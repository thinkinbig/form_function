import logging
import math
import os

import pandas as pd

from src import HJ
from src.DataTableObject import DataTableObject, NO_MARK, HIGHLIGHT_MARK, FONT_MARK
from src.CalculationHandler import CalculationHandler
from src.standard import TemperatureUnit, PressureUnit, FlowUnit, DensityUnit, MolecularWeightUnit, N2, N5
from src.exceptions import SteamFormatError, QueryBeforeCalculationError, NotAttachedException, FormatError, \
    ExtractError, ActualConditionError
from src.io import read_sheet_by_index, read_sheet_by_name
from src.utils import ignore_value, EmptyException, in_to_mm, is_number, float_to_percent_str, \
    float_rounding, extract_value, is_inch, replace_inch


class MainService:
    open_rate_keys = ['F_BEM_JSKDMIN', 'F_BEM_JSKDNOR', 'F_BEM_JSKDMAX']
    open_rate_flag_keys = ['F_BEM_JSKDPD1', 'F_BEM_JSKDPD2', 'F_BEM_JSKDPD3']
    cv_keys = ['F_BEM_QtyJSCV', 'F_BEM_QtyJSCVnro', 'F_BEM_QtyJSCVmax']
    cv_flag_keys = ['F_BEM_QtyJSCVPD1', 'F_BEM_QtyJSCVPD2', 'F_BEM_QtyJSCVPD3']
    error_flag_key = 'F_BEM_ROW_ERROR'
    liquid_speed_keys = ['F_BEM_QtyJSHKLS', 'F_BEM_QtyJSHKLSnro', 'F_BEM_QtyJSHKLSmax']
    # 工况密度
    actual_density_keys = ['F_BEM_GKMD1', 'F_BEM_GKMD2', 'F_BEM_GKMD3']
    noise_keys = ['F_BEM_JSZY', 'F_BEM_Text2', 'F_BEM_Text3']
    close_operation_key = 'F_BEM_QtyJSCZL'
    open_operation_key = 'F_BEM_KQCZL'
    adjustment_operation_key = 'F_BEM_TJCZL'

    def __init__(self):
        self.attached = False
        self.fl_xt_fd_filled = False

        self.dto = None
        self.provider = None

    def attach(self, provider) -> None:
        self.provider = provider
        self.on_attach()
        self.attached = True

    @property
    def task_size(self) -> int:
        """
        Get task size
        Returns
        -------
        int : task size
        """
        return self.dto.dataframe.shape[0] - 1

    def on_attach(self) -> None:
        self.dto = DataTableObject(self.provider.standard, self.provider.input)

    def cv_exists(self) -> bool:
        """
        Check if Cv exists
        Returns
        -------
        bool : True if exists, False otherwise
        """
        try:
            return not pd.isnull(self.rated_cv)
        except EmptyException:
            return False

    @ignore_value
    def insert_cv(self) -> None:
        """
        Calculate Cv
        """
        # 判断额定cv是否为空
        if self.cv_exists():
            return
        # 阀门型号
        # 阀座通径
        valve_d = self.d
        logging.debug(f"value_d: {valve_d}")
        # 阀门类型
        valve_type = self.valve_type
        # get first two characters of valve type
        if valve_type[:2] == 'M2':
            cv_sheet = read_sheet_by_name(self.provider.cv_path, 'M2', 1)
            # 读取就近的阀座通径行
            cv_row = cv_sheet.loc[(cv_sheet['FZTJ'] - valve_d).abs().argsort()[:1]]
            # 判断流量特性是否是等百分比的
            if 'L' in self.valve_type:
                # 读取线性列
                cv = cv_row['线性'].values[0]
            else:
                # 读取等百分比列
                cv = cv_row['等百分比'].values[0]
        else:
            # 读取额定cv
            logging.debug(f'valve type: {valve_type[:2]} valve d: {valve_d}')
            cv_sheet = read_sheet_by_name(self.provider.cv_path, valve_type[:2])
            cv = cv_sheet.loc[cv_sheet['阀座通径'].apply(lambda x: abs(x - valve_d)).idxmin(), '额定Cv']
        self.dto['F_BEM_EDCv'] = cv
        logging.debug(f"inserted cv: {cv}")

    @ignore_value
    def insert_fl_xt_fd(self) -> None:
        """
        Calculate fl, xt, xd
        Returns
        -------
        None

        """
        logging.debug(f"Start to calculate fl, xt, fd")
        characteristic_df = read_sheet_by_index(self.provider.valve_characteristic, 0)
        # 阀门型号
        valve_type = self.valve_type
        # 读取流向
        flow_direction = self.flow_direction
        match_df = self.get_fl_xt_xd(valve_type, flow_direction, characteristic_df)
        if match_df.empty:
            return
        assert match_df.shape[0] == 1
        fl = match_df['Fl'].values[0]
        xt = match_df['Xt'].values[0]
        fd = match_df['Fd'].values[0]
        gyktb = match_df['固有可调比'].values[0]
        self.dto['F_BEM_Fl'] = fl
        self.dto['F_BEM_XT'] = xt
        self.dto['F_BEM_Fd'] = fd
        self.dto['F_BEM_GYKTB1'] = gyktb
        self.fl_xt_fd_filled = True

    @staticmethod
    def get_fl_xt_xd(valve_type: str, flow_direction: str, characteristic_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get row serie with valve type and flow direction

        In sql:
        SELECT * FROM characteristic_df WHERE '型号' = valve_type AND '流向' = flow_direction

        Parameters
        ----------
        characteristic_df: pd.DataFrame of characteristic table
        valve_type : str of valve type
        flow_direction : str of flow direction

        Returns
        -------
        pd.DataFrame
        """
        # 匹配前两个字符

        data = characteristic_df.loc[(characteristic_df['型号'].map(lambda x: x[:2]) == valve_type[:2]) &
                                     (characteristic_df['流向'] == flow_direction)]
        return data.head(1)

    @ignore_value
    def insert_min_max(self) -> None:
        """
        计算cv值 喉口流速 开度和噪音
        结果写进 xxx_min xxx_max
        Returns
        -------
        None
        """
        medium_status = self.medium_status
        logging.debug(f"Start to calculate cv, medium status: {medium_status}")
        calculation = CalculationHandler(self)
        for i in range(3):
            try:
                kv = calculation.cal_cv_with_index(i)
                logging.debug(f"calculated kv: {kv} with index {i}")
                if kv is None:
                    continue
                cv = kv * 1.167
                self.dto[self.cv_keys[i]] = float_rounding(cv)
                if cv < self.rated_cv:
                    self.dto[self.cv_flag_keys[i]] = HIGHLIGHT_MARK ^ FONT_MARK
                elif cv < self.rated_cv * 0.15 or cv > self.rated_cv * 1.5:
                    self.dto[self.cv_flag_keys[i]] = HIGHLIGHT_MARK
            except SteamFormatError as e:
                logging.error(e)
                self.dto[self.cv_keys[i]] = str(e)
                # self.dto[self.cv_flag_keys[i]] = HIGHLIGHT_MARK ^ FONT_MARK
            except QueryBeforeCalculationError as e:
                logging.debug(e)
            except EmptyException as e:
                logging.debug(e)
        logging.debug(f"Finish to calculate cv")

        for i in range(3):
            try:
                open_rate = calculation.cal_open_rate_with_index(i)
                logging.debug(f"calculated open rate: {open_rate} with index {i}")
                if open_rate is None:
                    continue
                self.dto[self.open_rate_keys[i]] = float_to_percent_str(open_rate)
                if 0 < open_rate < 0.2 or 1 > open_rate > 0.7:
                    self.dto[self.open_rate_flag_keys[i]] = HIGHLIGHT_MARK
                elif open_rate < 0 or open_rate > 1:
                    self.dto[self.open_rate_flag_keys[i]] = HIGHLIGHT_MARK ^ FONT_MARK
            except QueryBeforeCalculationError as e:
                logging.error(e)
            except EmptyException as e:
                logging.error(e)
        logging.debug(f"Finish to calculate open rate")

        for i in range(3):
            try:
                velocity = calculation.cal_liquid_speed_with_index(i)
                if velocity is None:
                    continue
                self.dto[self.liquid_speed_keys[i]] = float_rounding(velocity)
            except ActualConditionError as e:
                logging.error(e)
            except QueryBeforeCalculationError as e:
                logging.error(e)
            except EmptyException as e:
                logging.error(e)
        logging.debug(f"Finish to calculate liquid speed")

        for i in range(3):
            try:
                noise = calculation.cal_noise_with_index(i)
                if noise is None:
                    continue
                if 0 < noise < 85:
                    self.dto[self.noise_keys[i]] = float_rounding(noise)
                else:
                    self.dto[self.noise_keys[i]] = "≤85"
            except QueryBeforeCalculationError as e:
                logging.error(e)
            except EmptyException as e:
                logging.error(e)
        logging.debug(f"Finish to calculate noise")

        valve_type = self.valve_type
        try:
            logging.debug(f"Start to calculate torque, valve type: {valve_type}")
            if valve_type.startswith('W1'):
                T = calculation.torque_w1()
                self.dto[self.open_operation_key] = float_rounding(T)
                self.dto[self.close_operation_key] = float_rounding(T)
            elif valve_type.startswith('W9'):
                w_F0, w_F1 = calculation.torque_w9()
                self.dto[self.close_operation_key] = float_rounding(w_F0)
                if w_F1 != 0:
                    self.dto[self.adjustment_operation_key] = float_rounding(w_F1)
            logging.debug(f"Finish to calculate torque")
            logging.debug(f"Start to calculate force, valve type: {valve_type}")
            if valve_type.startswith('M'):
                F_c, F_o = calculation.force_m()
                self.dto[self.close_operation_key] = float_rounding(F_c)
                if F_o != 0:
                    self.dto[self.open_operation_key] = float_rounding(F_o)
            elif valve_type.startswith('P'):
                F_c, F_o = calculation.force_p()
                self.dto[self.close_operation_key] = float_rounding(F_c)
                if F_o != 0:
                    self.dto[self.open_operation_key] = float_rounding(F_o)
            logging.debug(f"Finish to calculate force")
        except EmptyException as e:
            logging.debug(e)
        except QueryBeforeCalculationError as e:
            logging.error(e)
        except ValueError as e:
            self.dto[self.error_flag_key] = 1
            logging.error(e)

    @staticmethod
    def _pressure(value: float, unit: str) -> float:
        return PressureUnit.convert(value, unit)

    @staticmethod
    def _temperature(value: float, unit: str) -> float:
        return TemperatureUnit.convert(value, unit)

    @staticmethod
    def _flow(value: float, unit: str,
              p1: float | None = None,
              t1: float | None = None) -> float:
        return FlowUnit.convert(value, unit, p1, t1)

    @staticmethod
    def _density(value: float, unit: str):
        return DensityUnit.convert(value, unit)

    def process_end(self) -> bool:
        """
        Check if end of file
        Returns
        -------
        bool : True if end of file, False otherwise
        """
        return self.dto.eof

    @property
    def leaking_level(self):
        if not self.attached:
            raise NotAttachedException('leaking_level accessed before attached')
        return self.dto["F_BEM_Assistant11"]

    @property
    def flow_direction(self) -> str:
        if not self.attached:
            raise NotAttachedException('flow_direction accessed before attached')
        try:  # 读取流向
            return self.dto['F_BEM_LXComboXS']
        except EmptyException:
            return '流开'

    def is_blocked_flow(self, index):
        if not self.attached:
            raise NotAttachedException('is_blocked accessed before attached')
        return self.dto["F_BEM_ZSLPD" + str(index + 1)] == '是'

    @property
    def close_pressure(self) -> float:
        """

        关闭压差
        Returns
        -------
        float : 关闭压差
        """
        if not self.attached:
            raise NotAttachedException('Delta P accessed before attached')
        close_delta_p = self.dto['F_BEM_GBYC']
        if not is_number(close_delta_p):
            raise FormatError('Delta P is not a number')
        try:
            return self._pressure(float(close_delta_p), self.dto['F_BEM_GBYCDW'])
        except ValueError as e:
            logging.error(e)
            self.dto[self.error_flag_key] = 1

    @property
    def z1(self) -> float:
        """
        压缩系数
        """
        if not self.attached:
            raise NotAttachedException('Z1 accessed before attached')
        return self.dto['F_BEM_TextYSXS']

    @property
    def medium_status(self) -> str:
        """
        介质状态
        """
        if not self.attached:
            raise NotAttachedException('CalculationHandler accessed before attached')
        return self.dto['F_BEM_JZZT']

    @property
    def Pv(self) -> float:
        """
        蒸汽压力
        """
        if not self.attached:
            raise NotAttachedException('Pv accessed before attached')
        # return self.dto['F_BEM_ZQYL']
        return 0.701

    @property
    def standard_rho(self) -> float:
        """
        标况密度

        Returns
        -------
        float 标况密度
        """
        if not self.attached:
            raise NotAttachedException('Rho accessed before attached')
        try:
            # 读取标况密度
            rho = self.dto['F_BEM_ROH']
            rho_unit = self.dto['F_BEM_LLDWCOBOM11211']
            return self._density(rho, rho_unit)
        except EmptyException:
            # 如果标况密度为空，则计算标况密度
            num = 0
            rho = 0
            calculation = CalculationHandler(self)
            for i in range(3):
                s_rho = calculation.calculate_standard_rho(self.medium_status, self.actual_rho(i), self.t1(i),
                                                           self.z1, self.p1(i))
                if s_rho:
                    num += 1
                    rho += s_rho
            if num == 0:
                raise EmptyException('标况密度为空')
            else:
                return rho / num
        except ValueError as e:
            logging.error(e)
            self.dto[self.error_flag_key] = 1

    def actual_rho(self, index: int) -> float:
        """
        工况密度
        """
        if not self.attached:
            raise NotAttachedException('Rho accessed before attached')
        try:
            rho_unit = self.dto['F_BEM_LLDWCOBOM11211']
            rho = self.dto[self.actual_density_keys[index]]
            return self._density(rho, rho_unit)
        except ValueError as e:
            logging.error(e)
            self.dto[self.error_flag_key] = 1

    def q(self, index: int) -> float:
        """
        液体流量/气体工况流量 q
        """
        if not self.attached:
            raise NotAttachedException('Q accessed before attached')

        flow_unit = self.flow_unit

        keys = ['F_BEM_QtyYTLL', 'F_BEM_QtyYTLLNRO', 'F_BEM_QtyYTLLmax']

        v = self.dto[keys[index]]

        if not is_number(str(v)):
            raise FormatError('Q must be number')

        v = float(v)

        if self.medium_status == '气体' or self.medium_status == "饱和蒸汽":
            standard_rho = None
            actual_rho = None
            try:
                standard_rho = self.dto['F_BEM_ROH']
                actual_rho = self.dto[self.actual_density_keys[index]]
            except EmptyException:
                pass
            try: # 读取温度
                t1 = self.t1(index)
            except EmptyException:
                logging.debug(f"温度{index}为空")
                t1 = None
            try: # 读取压力
                p1 = self.p1(index)
            except EmptyException:
                logging.debug(f"压力{index}为空")
                p1 = None

            if flow_unit.is_standard_flow_unit():
                if standard_rho:
                    logging.info("工况条件 标况密度(F_BEM_ROH)，标况流量(F_BEM_QtyYTLL，F_BEM_QtyYTLLNRO，F_BEM_QtyYTLLmax)，"
                                 "流量单位带N")
                    rho = standard_rho
                elif actual_rho:
                    logging.info("工况条件 标况密度(F_BEM_ROH)，工况流量(F_BEM_QtyYTLL，F_BEM_QtyYTLLNRO，F_BEM_QtyYTLLmax)，"
                                 "流量单位带N")
                    rho = actual_rho
                else:
                    raise EmptyException('标况密度和工况密度都为空')
            else:
                if standard_rho:
                    logging.info(
                        "工况条件 标况密度(F_BEM_ROH)，标况流量(F_BEM_QtyYTLL，F_BEM_QtyYTLLNRO，F_BEM_QtyYTLLmax)")
                    rho = standard_rho
                elif actual_rho:
                    logging.info(
                        "工况条件 标况密度(F_BEM_ROH)，工况流量(F_BEM_QtyYTLL，F_BEM_QtyYTLLNRO，F_BEM_QtyYTLLmax)")
                    rho = actual_rho
                else:
                    raise EmptyException('标况密度和工况密度都为空')

            try:
                return FlowUnit.tune_flow(rho, self._flow(v, flow_unit, p1, t1), flow_unit)
            except AssertionError as e:
                logging.error("标况流量单位下，压力和温度不能为空")

    @property
    def cv_flag_array(self) -> list[int]:
        """
        Cv计算标志
        """
        if not self.attached:
            raise NotAttachedException('Cv_flag_array accessed before attached')
        return [self.dto["F_BEM_QtyJSCVPD" + str(i + 1)] for i in range(3)]

    @property
    def open_flag_array(self) -> list[int]:
        """
        开启标志
        """
        if not self.attached:
            raise NotAttachedException('Open_flag_array accessed before attached')
        return [self.dto["F_BEM_JSKDPD" + str(i + 1)] for i in range(3)]

    def p1(self, index: int):
        """
        阀前压力 p1
        """
        if not self.attached:
            raise NotAttachedException('P1 array accessed before attached')
        keys = ['F_BEM_QtyFQYLP1', 'F_BEM_QtyFQYLP1nro', 'F_BEM_QtyFQYLP1max']
        pressure_unit = self.dto['F_BEM_YLDWCOBOM1']
        if not is_number(str(self.dto[keys[index]])):
            raise FormatError('p1 must be number')
        try:
            return self._pressure(float(self.dto[keys[index]]), pressure_unit)
        except ValueError as e:
            logging.error(e)
            self.dto[self.error_flag_key] = 1

    def p2(self, index: int):
        """
        阀后压力 p2
        """
        if not self.attached:
            raise NotAttachedException('P2 array accessed before attached')
        keys = ['F_BEM_QtyFHYLP2', 'F_BEM_QtyFQYLP2nro', 'F_BEM_QtyFQTLP2max']
        pressure_unit = self.dto['F_BEM_LLDWCOBOM11']
        if not is_number(str(self.dto[keys[index]])):
            raise FormatError('p2 must be number')
        try:
            return self._pressure(float(self.dto[keys[index]]), pressure_unit)
        except ValueError as e:
            logging.error(e)
            self.dto[self.error_flag_key] = 1

    def t1(self, index: int):
        if not self.attached:
            raise NotAttachedException('T1 array accessed before attached')
        keys = ['F_BEM_QtyCZWD', 'F_BEM_QtyCZWDnro', 'F_BEM_QtyCZWDmax']
        temperature_unit = self.dto['F_BEM_WDDW']
        if not is_number(str(self.dto[keys[index]])):
            raise FormatError('t1 must be number')
        try:
            return self._temperature(float(self.dto[keys[index]]), temperature_unit)
        except ValueError as e:
            logging.error(e)
            self.dto[self.error_flag_key] = 1

    @property
    def Ff(self) -> float:
        if not self.attached:
            raise NotAttachedException('Ff accessed before attached')
        try:
            return 0.96 - 0.28 * (math.sqrt(self.Pv / self.Pc))
        except EmptyException:
            return 0.96
        except ValueError as e:
            logging.error(e)
            self.dto[self.error_flag_key] = 1

    def c(self, index: int) -> float:
        """
        cv系数
        """
        if not self.attached:
            raise NotAttachedException('C accessed before attached')
        if not is_number(str(self.dto[self.cv_keys[index]])):
            raise FormatError('C must be number')
        return self.dto[self.cv_keys[index]]

    @property
    def filling_material(self) -> str:
        """
        填料
        Returns
        -------

        """
        if not self.attached:
            raise NotAttachedException('filling_material accessed before attached')
        try:  # 读取填料
            return self.dto['F_BEM_TLXS']
        except EmptyException:
            return 'PTFE'

    def k(self, index: int) -> float:
        """
        kv系数
        kv = cv / 1.167
        Returns
        -------

        """
        if not self.attached:
            raise NotAttachedException('kv accessed before attached')
        return self.c(index) / 1.167

    @property
    def V(self) -> list[float]:
        """
        流量速度
        """
        if not self.attached:
            raise NotAttachedException('v accessed before attached')
        return [
            self.dto['F_BEM_QtyJSHKLS'],
            self.dto['F_BEM_QtyJSHKLSnro'],
            self.dto['F_BEM_QtyJSHKLSmax'],
        ]

    def v(self, index: int) -> float:
        """
        流量速度
        """
        if not self.attached:
            raise NotAttachedException('v accessed before attached')
        keys = ['F_BEM_QtyJSHKLS', 'F_BEM_QtyJSHKLSnro', 'F_BEM_QtyJSHKLSmax']
        if not is_number(str(self.dto[keys[index]])):
            raise FormatError('v must be number')
        return self.dto[keys[index]]

    def delta_p_chocked(self, index: int) -> float:
        return self.Fl * self.Fl * (self.p1(index) - self.Ff * self.Pv)

    def delta_p_sizing(self, index: int) -> float:
        Delta_P_Chocked = self.delta_p_chocked(index)
        P1 = self.p1(index)
        P2 = self.p2(index)
        if Delta_P_Chocked < (P1 - P2):
            return Delta_P_Chocked
        else:
            return P1 - P2

    @property
    def valve_type(self) -> str:
        """
        阀门型号
        """
        if not self.attached:
            raise NotAttachedException('Valve type accessed before attached')
        return self.dto['F_BEM_FMXH1XS']

    @property
    def flow_unit(self) -> FlowUnit:
        """
        流量单位
        Returns
        -------
        FlowUnit

        """
        if not self.attached:
            raise NotAttachedException('Liquid unit accessed before attached')
        unit = self.dto['F_BEM_LLDWCOBOM']
        return FlowUnit(unit)

    @property
    def d(self) -> float:
        """
        控制阀座通径
        Returns
        -------

        """
        if not self.attached:
            raise NotAttachedException('D accessed before attached')
        try:
            return self.dto['F_BEM_FZTJ']
        except EmptyException:
            # 若没有直接提供阀座通径的尺寸数据，则读取 F_BEM_GCTJXS 公称通径销售的数据。
            gctjxs = str(self.dto['F_BEM_GCTJXS'])
            try:
                d = self.get_valid_d(self.valve_type, gctjxs)
                d = float(d)
                # 保存到dto中
                self.dto['F_BEM_FZTJ'] = d
                return d
            except ExtractError:
                self.dto[self.error_flag_key] = 1
                raise EmptyException('阀座通径数据格式错误')
            except ValueError as e:
                logging.error(e)
                self.dto[self.error_flag_key] = 1

    @staticmethod
    def get_valid_d(valve_type: str, gctjxs: str) -> float:
        """
        获取有效的阀座通径
        Returns
        -------

        """

        gctj, sj = extract_value(valve_type)
        if is_inch(gctjxs):
            tj = float(replace_inch(gctjxs)) * 25.4
            if abs(gctj / tj) < 20:
                return sj
            else:
                return sj * 25.4
        elif is_number(gctjxs):
            return float(gctjxs)
        elif gctjxs.startswith('DN'):
            return float(gctjxs[2:])
        else:
            raise FormatError('阀座通径数据格式错误')

    @property
    def D(self) -> float:
        """
        管道外径
        Returns
        -------

        """
        if not self.attached:
            raise NotAttachedException('D accessed before attached')
        return in_to_mm(str(self.dto['F_BEM_TextGDWJ']))

    @property
    def rated_cv(self) -> float:
        """
        额定 cv
        """
        if not self.attached:
            raise NotAttachedException('Cv accessed before attached')
        return self.dto["F_BEM_EDCv"]

    @property
    def Fl(self) -> float:
        """
        无附接管件控制阀的液体压力恢复系数 Fl
        """
        if not self.attached:
            raise NotAttachedException('Fl accessed before attached')
        if not self.fl_xt_fd_filled:
            raise QueryBeforeCalculationError('Fl accessed before Fl filled')
        return self.dto["F_BEM_Fl"]

    @property
    def xT(self) -> float:
        """
        阻塞流条件下无附接管件控制阀的压差比系数
        """
        if not self.attached:
            raise NotAttachedException('xT accessed before attached')
        if not self.fl_xt_fd_filled:
            raise QueryBeforeCalculationError('xT accessed before xt filled')
        return self.dto["F_BEM_XT"]

    @property
    def Fd(self) -> float:
        """
        控制阀类型修正系数 Fd
        """
        if not self.attached:
            raise NotAttachedException('Fd accessed before attached')
        if not self.fl_xt_fd_filled:
            raise QueryBeforeCalculationError('Fd accessed before Fd filled')
        return self.dto["F_BEM_Fd"]

    @property
    def R(self) -> float:
        """
        再热系数
        """
        if not self.attached:
            raise NotAttachedException('R accessed before attached')
        return self.dto["F_BEM_GYKTB1"]

    @property
    def M(self) -> float:
        """
        流体分子量 M
        """
        if not self.attached:
            raise NotAttachedException('M accessed before attached')
        try:
            m = self.dto["F_BEM_MOLZL"]
        except EmptyException:
            logging.debug(f"standard rho: {self.standard_rho}")
            m = self.standard_rho * 22.4
        try:
            unit = self.dto["F_BEM_ZYDW1"]
        except EmptyException:
            unit = ''
        if not is_number(m):
            raise FormatError('M is not a number')
        try:
            return MolecularWeightUnit.convert(float(m), unit)
        except ValueError as e:
            logging.error(e)
            self.dto[self.error_flag_key] = 1

    @property
    def Fk(self) -> float:
        """
        比热系数 Fk
        Returns
        -------

        """
        if not self.attached:
            raise NotAttachedException('Fk accessed before attached')
        try:
            return self.dto['F_BEM_BRBXS']
        except EmptyException:
            return 1.4

    @property
    def Pc(self) -> float:
        """
        绝对热力学临界压力 Pc
        """
        if not self.attached:
            raise NotAttachedException('Pc accessed before attached')
        value = self.dto['F_BEM_TextLJYL']
        if not is_number(value):
            raise FormatError('Pc is not a number')
        value = float(value)
        unit = self.dto['F_BEM_LLDWCOBOM1121']
        return self._pressure(value, unit)

    @property
    def Tc(self) -> float:
        """
        绝对热力学临界温度 Tc
        """
        if not self.attached:
            raise NotAttachedException('Tc accessed before attached')
        return self.dto['F_BEM_RLXLJWD']

    @property
    def Fp(self) -> float:
        """
        计算Fp
        Returns
        -------
        float
        """
        Fp = HJ.equation15_version1(self.zeta, N2, self.rated_cv, self.d)
        return Fp

    @property
    def zeta(self) -> float:
        """
        计算阻塞流条件下的阻力系数
        Returns
        -------
        float
        """
        return 1.5 * ((1 - (self.d / self.D) ** 2) ** 2)

    @property
    def zeta_1(self) -> float:
        """
        计算阻塞流条件下的阻力系数
        Returns
        -------
        float
        """
        return 0.5 * ((1 - (self.d / self.D) ** 2) ** 2)

    @property
    def zeta_2(self) -> float:
        """
        计算阻塞流条件下的阻力系数
        Returns
        -------
        float
        """
        return (1 - (self.d / self.D) ** 2) ** 2

    @property
    def x_choked(self) -> float:
        """
        计算阻塞流条件下的压差比系数
        Returns
        -------
        float
        """
        xtp = HJ.equation_xtp(self.xT, self.Fp, self.zeta_1, N5, self.rated_cv, self.d)
        return self.Fk * xtp

    def x_sizing(self, index) -> float:
        """
        计算x
        Parameters
        ----------
        index : int
            0, 1, 2
        Returns
        -------
        float
        """
        logging.debug(f"index: {index}")
        logging.debug(f"p1: {self.p1(index)}")
        logging.debug(f"p2: {self.p2(index)}")
        x = (self.p1(index) - self.p2(index)) / self.p1(index)
        logging.debug(f"x: {x}")
        logging.debug(f"x_choked: {self.x_choked}")
        if x > self.x_choked:
            return self.x_choked
        else:
            return x

    def y(self, index) -> float:
        """
        计算y
        Parameters
        ----------
        index : int
            0, 1, 2
        Returns
        -------
        float
        """
        return 1 - self.x_sizing(index) / 3 / self.x_choked

    def open(self, index) -> float:
        """
        计算开度
        Parameters
        ----------
        index : int
            0, 1, 2
        Returns
        -------
        float
        """
        c = self.c(index)
        cv = self.rated_cv
        R = self.R

        if "L" in self.valve_type:
            _open = HJ.equation_open_rate_l_value(c, cv, R)
        else:
            _open = HJ.equation_open_rate_e_value(c, cv, R)
        return _open

    def process(self) -> None:
        """
        Process all calculation
        """
        self.on_process_start()
        self.insert_cv()
        self.insert_fl_xt_fd()
        self.insert_min_max()

    def next(self) -> None:
        """
        Reset all flags and move to next row
        Returns
        -------

        """

        self.dto.next()

    def on_process_start(self) -> None:
        # set extra values
        for i in range(3):
            self.dto[self.open_rate_flag_keys[i]] = NO_MARK
            self.dto[self.cv_flag_keys[i]] = NO_MARK
            self.dto[self.error_flag_key] = NO_MARK

    def export(self, output_path: str) -> None:
        """
        Export to excel
        if output file exists, remove it first
        Parameters
        ----------
        output_path : str
            output file path
        """
        if os.path.exists(output_path):
            os.remove(output_path)
        self.dto.export_dataframe(output_path)
