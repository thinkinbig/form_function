import logging
import math
import re
import typing

import pandas as pd

from src import HJ
from src.exceptions import SteamFormatError, MediumStatusError, LeakingLevelError, EmptyException
from src.io import read_sheet_by_index
from src.standard import KELVIN_CONST, N8, N9, N6, N1
from src.utils import special_roman_to_int, ignore_value

if typing.TYPE_CHECKING:
    from src.service import MainService


class CalculationHandler:
    LIQUID = "液体"
    GAS = "气体"
    VAPOUR = "饱和蒸汽"
    EMPTY = ""

    def __init__(self, service_instance: 'MainService'):
        self.service_instance = service_instance

    @ignore_value
    def cal_cv_with_index(self, index: int) -> float:
        service = self.service_instance
        q = service.q(index)
        Fp = service.Fp
        p1 = service.p1(index)
        if service.medium_status == self.LIQUID:
            p2 = service.p2(index)
            delta_p_sizing = service.delta_p_sizing(index)
            delta_p_choked = service.delta_p_chocked(index)
            rho = service.rho
            if delta_p_choked < (p1 - p2):
                service.dto["F_BEM_ZSLPD" + str(index + 1)] = "是"
            else:
                service.dto["F_BEM_ZSLPD" + str(index + 1)] = "否"
            kv = HJ.equation1_value(q, N1, Fp, delta_p_sizing, rho)
        elif service.medium_status == self.GAS:
            y = service.y(index)
            x_sizing = service.x_sizing(index)
            p2 = service.p2(index)
            m = service.M
            t1 = service.t1(index)
            z1 = service.z1
            x = (p1 - p2) / p1
            if x > service.x_choked:
                service.dto["F_BEM_ZSLPD" + str(index + 1)] = '是'
            else:
                service.dto["F_BEM_ZSLPD" + str(index + 1)] = '否'
            if service.flow_unit == 'kg/h':
                logging.debug(f"q: {q} N8: {N8} Fp: {Fp} p1: {p1} y: {y} x_sizing: {x_sizing} m: {m} t1: {t1} z1: {z1}")
                kv = HJ.equation_gas_cv_w_value(q, N8, Fp, p1, y, x_sizing, m, t1, z1)
            else:
                kv = HJ.equation_gas_cv_q_value(q, N9, Fp, p1, y, x_sizing, m, t1, z1)
        elif service.medium_status == self.VAPOUR:
            y = service.y(index)
            x_sizing = service.x_sizing(index)
            rho = service.rho
            if service.flow_unit == 'kg/h':
                kv = HJ.equation_steam_value(q, N6, Fp, p1, y, x_sizing, rho)
            else:
                raise SteamFormatError('蒸汽计算时，流量单位必须是kg/h')
        elif service.medium_status == self.EMPTY:
            raise EmptyException('介质状态为空')
        else:
            raise MediumStatusError('介质状态错误')
        return kv

    @ignore_value
    def cal_noise_with_index(self, index: int) -> float:
        service = self.service_instance
        kv = service.k(index)
        p1 = service.p1(index)
        p2 = service.p2(index)
        if service.medium_status == self.LIQUID:
            if service.is_blocked_flow(index):
                noise = HJ.equation_noise_zsl(kv, p1, service.Pv, service.rho, p2)
            else:
                noise = HJ.equation_noise(kv, p1, service.Pv, service.rho, p2)
        else:
            noise = HJ.equation_noise_qt_value(kv, p1, service.t1(index), service.rho, p2)
        return noise

    @ignore_value
    def cal_liquid_speed_with_index(self, index: int) -> float:
        service = self.service_instance
        q = service.q(index)
        d = service.d
        rho = service.rho
        v = HJ.equation_liquid_speed_value(d, q)
        t1 = service.t1(index)
        p1 = service.p1(index)
        logging.debug(f"q: {q} d: {d} rho: {rho} v: {v} t1: {t1} p1: {p1}")
        if service.medium_status == self.LIQUID:
            return v
        else:
            return v * 1.01 * t1 / KELVIN_CONST / p1 * 1.5

    @ignore_value
    def cal_open_rate_with_index(self, index: int) -> float:
        service = self.service_instance
        open_rate = service.open(index)
        q = service.q(index)

        if open_rate <= 0 or q <= 0:
            return 0
        else:
            return open_rate

    def torque_w1(self) -> float:
        service_instance = self.service_instance
        d = service_instance.d
        row = self.get_row_by_range()
        if row.empty:
            raise EmptyException('DN不在范围内')
        A = row[2]
        T = HJ.equation_torque_w1(service_instance.close_pressure, d, A)
        return T

    def get_row_by_range(self) -> pd.Series:
        service_instance = self.service_instance
        torque_w1_df = read_sheet_by_index(service_instance.gui.operation, 2)
        d = service_instance.d
        for index, row in torque_w1_df.iterrows():
            if int(row[0][2:]) <= d < int(row[1][2:]):
                logging.debug(f"from: {row[0][2:]} to: {row[1][2:]}")
                return row
        return pd.Series()

    def torque_w9(self) -> tuple[float, float]:
        i = 4
        service_instance = self.service_instance
        pressureStr = service_instance.dto['F_BEM_GCYLXS']
        pressure = float(re.sub("[^0-9]+", "", pressureStr))
        if "PN" in pressureStr:
            if pressure > 25:
                i = 5
        elif "CL" in pressureStr:
            if pressure == 300:
                i = 5
        row = self.get_row_by_d(i)
        if row.empty:
            raise EmptyException('d不在范围内')
        w_d = row[2]
        w_D = row[3]
        w_W = row[4]
        w_H = row[12]
        w_a = row[16]
        tl = service_instance.filling_material
        if tl == "四氟":
            w_N = 0.7875
            w_M2 = 0.0203
        else:
            w_N = 1
            w_M2 = 0.0341
        # 轴承扭距
        F1 = w_d / 20 * 0.02 * (math.pi / 4 * w_D * w_D * service_instance.close_pressure / 98 + w_W) * 9.8 / 100
        # 填料扭距
        F2 = math.pi * w_d * w_d * w_H * w_N * w_M2 * 9.8 * 150 / 2000 / 100
        # 偏心扭距
        F3 = math.pi / 4 * w_D * w_D * w_a * service_instance.close_pressure * 100 * 9.8 / 100000 / 98
        # 阀座预紧扭距
        F4 = math.pi * w_D * w_D * 0.1 * 9.8 / 1000
        # 不平衡扭距
        w_p = []
        for i in range(3):
            try:
                w_p.append(service_instance.p1(i) - service_instance.p2(i) * 100 / 98)
            except EmptyException:
                pass
        w_F0 = F1 + F2 + F3 + F4 / 0.7
        if len(w_p) == 0:
            return w_F0, 0
        else:
            F5 = 1.5 * (0.07 * (w_D / 10) ** 3 + 4.71 / 100 * w_D * w_a / 10) * max(w_p) / 98 * 9.8

            w_F1 = F2 + F3 + F4 + F5 / 0.7

            return w_F0, w_F1

    def get_row_by_d(self, i: int) -> pd.Series:
        service_instance = self.service_instance
        torque_m9_df = read_sheet_by_index(service_instance.gui.operation, i, skiprows=1)
        d = service_instance.d
        for index, row in torque_m9_df.iterrows():
            if int(row[0]) == d:
                return row
        return pd.Series()

    def force_p(self) -> tuple[float, float]:
        service_instance = self.service_instance
        force_p_df = read_sheet_by_index(service_instance.gui.operation, 1, skiprows=1)
        row = self.get_nearest_searching_row(force_p_df)
        ds = row[1]
        dg = row[2]
        L = row[4]
        FY = self.get_FY(row)
        # 填料摩擦力
        a = 0.0388
        if service_instance.filling_material == '四氟':
            a = 0.02
        tlmcl = math.pi * ds * L * 4 * a
        if service_instance.flow_direction == '流关':
            FUnbalance = ds * ds * math.pi * service_instance.close_pressure / 40
        else:
            FUnbalance = dg * dg * math.pi * service_instance.close_pressure / 40
        F_c = tlmcl + FUnbalance + FY
        pp = []
        for i in range(3):
            try:
                pp.append(service_instance.p1(i) - service_instance.p2(i))
            except EmptyException:
                pass
        if len(pp) == 0:
            F_o = 0
        else:
            F_o = math.pi / 4 * ds * ds * max(pp)
        return F_c, F_o

    def force_m(self) -> tuple[float, float]:
        service_instance = self.service_instance
        force_m_df = read_sheet_by_index(service_instance.gui.operation, 0, skiprows=1)
        row = self.get_nearest_searching_row(force_m_df)
        P1_array = []
        P2_array = []
        for i in range(3):
            try:
                P1_array.append(service_instance.p1(i))
            except EmptyException:
                pass
            try:
                P2_array.append(service_instance.p2(i))
            except EmptyException:
                pass
        d = service_instance.d
        ds = row[1]
        dg = row[2]
        dp = row[3]
        L = row[4]
        H = row[15]
        FY = self.get_FY(row)
        # 填料摩擦力
        a = 0.0388
        if service_instance.filling_material == '四氟':
            a = 0.02
        tlmcl = math.pi * ds * L * 4 * a
        if service_instance.flow_direction == '流关':
            FUnbalance = math.pi / 4 * (dg / 10 + 0.05) ** 2 * max(P1_array)
        else:
            FUnbalance = math.pi / 4 * (ds / 10) ** 2 - (dg / 10 + 0.05) ** 2 * max(P1_array)
        if service_instance.filling_material == '四氟':
            FH = 1.5 * dp / 10 * d / 25.4
        else:
            FH = math.pi * dp * H * 1.582 * 0.15
        F_c = tlmcl + FUnbalance + FY + FH
        pp = []
        for i in range(3):
            try:
                pp.append(service_instance.p1(i) - service_instance.p2(i))
            except EmptyException:
                pass
        if len(pp) == 0:
            F_o = 0
        else:
            F_o = math.pi / 4 * ds * ds * max(pp)
        return F_c, F_o

    def get_nearest_searching_row(self, df_to_search: pd.DataFrame) -> list:
        valve_d = self.service_instance.d

        # 就近查找
        row = df_to_search.loc[(df_to_search['DN'] - valve_d).abs().argsort()[:1]]
        row = row.values.tolist()[0]
        return row

    def get_FY(self, row):
        leaking_level = special_roman_to_int(self.service_instance.leaking_level)
        if leaking_level == 3:
            FY = row[5]
        elif leaking_level == 4:
            FY = row[6]
        elif leaking_level == 5:
            FY = row[7]
        elif leaking_level == 6:
            FY = row[8]
        else:
            raise LeakingLevelError('泄漏等级错误!')
        return FY
