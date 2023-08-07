import logging
import math

from src.standard import FlowUnit


def equation_noise(kv: float, P1: float, Pv: float, Roh: float, P2: float) -> float:
    """
    Calculate the noise of a fluid in non-blocked flow.
    计算非堵塞流体的噪声。

    Parameters:
        kv (float): Flow coefficient.
        P1 (float): Inlet pressure.
        Pv (float): Saturated vapor pressure.
        Roh (float): Liquid density.
        P2 (float): Outlet pressure.

    Returns:
        float: Noise value in decibels.
    """
    noise = 10 * math.log10(kv) + 18 * math.log10(P1 - Pv) - 5 * math.log10(Roh) \
            + 18 * math.log10((P1 - P2) / (P1 - Pv) / 0.25) + 40
    return noise


def equation_noise_zsl(kv: float, P1: float, Pv: float, Roh: float, P2: float) -> float:
    """
    Calculate the noise of a fluid in blocked flow.
    计算堵塞流体的噪声。

    Parameters:
        kv (float): Flow coefficient.
        P1 (float): Inlet pressure.
        Pv (float): Saturated vapor pressure.
        Roh (float): Liquid density.
        P2 (float): Outlet pressure.

    Returns:
        float: Noise value in decibels.
    """
    noise = 10 * math.log10(kv) + 18 * math.log10(P1 - Pv) - 5 * math.log10(Roh) + \
            29 * math.log10((P1 - P2) / (P1 - Pv) - 0.5) * 0.75 - (268 + 19) * ((P1 - P2) / (P1 - Pv) - 0.5) + 40
    return noise


def equation_noise_qt(kv: list[float], P1: list[float], T1: list[float], Roh: float, P2: list[float]) -> list[float]:
    """
    没什么卵用的函数，不知道是干嘛的
    Parameters:
        kv (list[float): Flow coefficients.
        P1 (list[float]): Inlet pressures.
        T1 (list[float]): Inlet temperatures.
        Roh (float): Liquid density.
        P2 (list[float]): Outlet pressures.

    Returns:
        list: Noise values in decibels.
    """
    noise = [0.0] * 3
    for i in range(3):
        noise[i] = 14 * math.log10(kv[i]) + 18 * math.log10(P1[i]) + 5 * math.log10(T1[i]) - 5 * math.log10(Roh) \
                   + 20 * math.log10(P1[i] / P2[i]) + 52

    return noise


def equation_noise_qt_value(kv: float, P1: float, T1: float, Roh: float, P2: float) -> float:
    """
    没什么卵用的函数，不知道是干嘛的
    Parameters:
        kv (float): Flow coefficient.
        P1 (float): Inlet pressure.
        T1 (float): Inlet temperature.
        Roh (float): Liquid density.
        P2 (float): Outlet pressure.

    Returns:
        float: Noise value in decibels.
    """
    noise = 14 * math.log10(kv) + 18 * math.log10(P1) + 5 * math.log10(T1) - 5 * math.log10(Roh) \
            + 20 * math.log10(P1 / P2) + 52

    return noise


def equation_steam(Q: list[float], N6: float, Fp: float, P1: list[float], Y: list[float], X_Sizing: list[float],
                   Roh: float) -> list[float]:
    """
    Calculate the saturated steam cv.
    计算饱和蒸汽的cv值。

    Parameters:
        Q (list[float]): Flow rates.
        N6 (float): Constant value N6.
        Fp (float): Constant value Fp.
        P1 (list[float]): Inlet pressures.
        Y (list[float]): Constant values Y.
        X_Sizing (list[float]): Sizing factors.
        Roh (float): Liquid density.

    Returns:
        list[float]: cv values.
    """
    cv = [0.0] * 3
    for i in range(3):
        cv[i] = Q[i] / N6 / Fp / Y[i] / (math.sqrt(X_Sizing[i] * P1[i] * Roh))

    return cv


def equation_steam_value(Q: float, N6: float, Fp: float, P1: float, Y: float, X_Sizing: float, Roh: float) -> float:
    """
    Calculate the saturated steam cv.
    计算饱和蒸汽的cv值。

    Parameters:
        Q (float): Flow rate.
        N6 (float): Constant value N6.
        Fp (float): Constant value Fp.
        P1 (float): Inlet pressure.
        Y (float): Constant value Y.
        X_Sizing (float): Sizing factor.
        Roh (float): Liquid density.

    Returns:
        float: cv value.
    """
    cv = Q / N6 / Fp / Y / (math.sqrt(X_Sizing * P1 * Roh))

    return cv


def equation_torque_w1(Delta_P: float, d: float, A: float) -> float:
    """
    Calculate torque using formula W1.
    使用公式W1计算扭矩。

    Parameters:
        Delta_P (float): Pressure difference.
        d (float): Diameter.
        A (float): Constant value A.

    Returns:
        float: Torque value.
    """
    T = 10.5 * Delta_P * d ** 3 * 1e-6 + A
    return T


def equation_liquid_speed(d: float, Q: list, unit: str, Roh: float) -> list:
    """
    Calculate throat velocity.
    计算喉口速度。

    Parameters:
        d (float): Throat diameter.
        Q (list): Flow rates.
        unit (str): Unit of flow rate ("kg/h" or other).
        Roh (float): Liquid density.

    Returns:
        list: Throat velocity values.
    """
    v = [0.0] * 3
    if unit == "kg/h":
        for i in range(3):
            v[i] = 1000000 * Q[i] / (math.pi * d * d) * 4 / Roh / 3600
    else:
        for i in range(3):
            v[i] = 1000000 * Q[i] / (math.pi * d * d) * 4 / 3600

    return v


def equation_liquid_speed_value(d: float, Q: float) -> float:
    """
    Calculate throat velocity.
    计算喉口速度。

    Parameters:
        d (float): Throat diameter.
        Q (float): Flow rate.
        unit (FlowUnit): Unit of flow rate.
        Roh (float): Liquid density.

    Returns:
        float: Throat velocity value.
    """
    v = 1000000 * Q / (math.pi * d * d) * 4 / 3600

    return v


def equation_gas_cv_q(Q: list, N9: float, Fp: float, P1: list, Y: list, X_Sizing: list, M: float, T1: list,
                      Z1: float) -> list:
    """
    Calculate gas volume flow rate.
    计算气体体积流量。

    Parameters:
        Q (list): Flow rates.
        N9 (float): Constant value N9.
        Fp (float): Constant value Fp.
        P1 (list): Inlet pressures.
        Y (list): Constant values Y.
        X_Sizing (list): Sizing factors.
        M (float): Molecular weight.
        T1 (list): Inlet temperatures.
        Z1 (float): Compressibility factor.

    Returns:
        list: cv values.
    """
    cv = [0.0] * 3
    for i in range(3):
        cv[i] = Q[i] / (N9 * Fp * P1[i] * Y[i] * math.sqrt(X_Sizing[i] / M / T1[i] / Z1))
    return cv


def equation_gas_cv_q_value(Q: float, N9: float, Fp: float, P1: float, Y: float, X_Sizing: float, M: float, T1: float,
                            Z1: float) -> float:
    """
    Calculate gas volume flow rate.
    Parameters
    ----------
        Q (list): Flow rates.
        N9 (float): Constant value N9.
        Fp (float): Constant value Fp.
        P1 (float): Inlet pressures.
        Y (float): Constant values Y.
        X_Sizing (float): Sizing factors.
        M (float): Molecular weight.
        T1 (float): Inlet temperatures.
        Z1 (float): Compressibility factor.

    Returns
    -------

    """
    cv = Q / (N9 * Fp * P1 * Y * math.sqrt(X_Sizing / M / T1 / Z1))
    return cv


def equation_gas_cv_w(Q: list, N8: float, Fp: float, P1: list, Y: list[float], X_Sizing: list[float], M: float,
                      T1: list[float], Z1: float) -> list:
    """
    Calculate gas mass flow rate.
    计算气体质量流量。

    Parameters:
        Q (list): Flow rates.
        N8 (float): Constant value N8.
        Fp (float): Constant value Fp.
        P1 (list[float]): Inlet pressures.
        Y (list[float]): Constant values Y.
        X_Sizing (list[float]): Sizing factors.
        M (float): Molecular weight.
        T1 (list[float]): Inlet temperatures.
        Z1 (float): Compressibility factor.

    Returns:
        list: cv values.
    """
    cv = [0.0] * 3
    for i in range(3):
        cv[i] = Q[i] / (N8 * Fp * P1[i] * Y[i] * math.sqrt(X_Sizing[i] * M / T1[i] / Z1))

    return cv


def equation_gas_cv_w_value(Q: float, N8: float, Fp: float, P1: float, Y: float, X_Sizing: float, M: float,
                            T1: float, Z1: float) -> float:
    """
    Calculate gas mass flow rate.
    Parameters
    ----------
        Q (list): Flow rates.
        N8 (float): Constant value N8.
        Fp (float): Constant value Fp.
        P1 (float): Inlet pressures.
        Y (float): Constant values Y.
        X_Sizing (float): Sizing factors.
        M (float): Molecular weight.
        T1 (float): Inlet temperatures.
        Z1 (float): Compressibility factor.

    Returns
    -------

    """
    cv = Q / (N8 * Fp * P1 * Y * math.sqrt(X_Sizing * M / T1 / Z1))
    return cv


def equation_xtp(xt: float, Fp: float, Zeta1: float, N5: float, edcv: float, d: float) -> float:
    """
    Calculate XTP value.
    计算XTP值。

    Parameters:
        xt (float): Xt value.
        Fp (float): Constant value Fp.
        Zeta1 (float): Zeta1 value.
        N5 (float): Constant value N5.
        edcv (float): EDCV value.
        d (float): Diameter.

    Returns:
        float: XTP value.
    """
    xtp = xt / Fp / Fp / (1 + (xt * Zeta1 / N5) * ((edcv / d / d) ** 2))
    return xtp


def equation_open_rate_e(C: list[float], edcv: float, R: float) -> list[float]:
    """
    Calculate open rate formula (equal percentage).
    计算开度（等百分比）。

    Parameters:
        C (list[float]): C values.
        edcv (float): EDCV value.
        R (float): R value.

    Returns:
        list[float]: Open rate values.
    """
    Open = [0.0] * 3
    for i in range(3):
        Open[i] = math.log10(C[i] / edcv) / math.log10(R) + 1

    return Open


def equation_open_rate_e_value(C: float, edcv: float, R: float) -> float:
    """
    Calculate open rate formula (equal percentage).
    计算开度（等百分比）。

    Parameters:
        C (float): C values.
        edcv (float): EDCV value.
        R (float): R value.

    Returns:
        float: Open rate values.
    """
    logging.debug(f"Open Rate E C {C} edcv {edcv} R {R}")
    Open = math.log10(C / edcv) / math.log10(R) + 1

    return Open


def equation_open_rate_l(C: list[float], edcv: float, R: float) -> list[float]:
    """
    Calculate open rate formula (linear).
    计算开度（线性）。

    Parameters:
        C (list[float]): C values.
        edcv (float): EDCV value.
        R (float): R value.

    Returns:
        list[float]: Open rate values.
    """
    Open = [0.0] * 3
    for i in range(3):
        Open[i] = (C[i] / edcv - 1 / R) / (1 - 1 / R)

    return Open


def equation_open_rate_l_value(C: float, edcv: float, R: float) -> float:
    """
    Calculate open rate formula (linear).
    计算开度（线性）。

    Parameters:
        C (float): C values.
        edcv (float): EDCV value.
        R (float): R value.

    Returns:
        float: Open rate values.
    """
    Open = (C / edcv - 1 / R) / (1 - 1 / R)

    return Open


def equation_4(d: float, D: float) -> float:
    """
    Formula 4: Zeta_1 + Zeta_2 = 1.5 * (1 - (d/D)^2)^2

    Parameters:
        d (float): d value.
        D (float): D value.

    Returns:
        float: Result of the formula.
    """
    return math.pow(1 - math.pow(d / D, 2), 2) * 1.5


def equation15_version1(Zeta: float, N_2: float, C_def: float, d: float) -> float:
    """
    Formula 15: F_P = 1 / sqrt(1 + (sum(Zeta) / N_2) * ((C_def / d^2)^2))

    Parameters:
        Zeta (float): Zeta value.
        N_2 (float): N_2 value.
        C_def (float): C_def value.
        d (float): d value.

    Returns:
        float: Result of the formula.
    """
    return 1 / (math.sqrt(1 + (Zeta / N_2) * (math.pow(C_def / d / d, 2))))


def equation1_value(Q: float, N_1: float, F_P: float, Delta_P_Sizing: float, Ro_rel: float) -> float:
    """
    Formula 1: C = Q / (N_1 * F_P * sqrt(Delta_P_Sizing / Ro_rel))

    Parameters:
        Q (float): Q value.
        N_1 (float): N_1 value.
        F_P (float): F_P value.
        Delta_P_Sizing (float): Delta_P_Sizing value.
        Ro_rel (float): Ro_rel value.

    Returns:
        float: Result of the formula.
    """
    # return Q / (N_1 * F_P * (math.sqrt(Delta_P_Sizing / Ro_rel)))
    return Q * (math.sqrt((Ro_rel / 1000) / Delta_P_Sizing)) / N_1 / F_P



def equation2_value(P2: float, P1: float) -> float:
    """
    Formula 2: Delta_P_Sizing = P2 - P1

    Parameters:
        P2 (float): P2 value.
        P1 (float): P1 value.

    Returns:
        float: Delta_P_Sizing value.
    """
    return P2 - P1





def equation3_value(C: float, N_1: float, F_P: float, Delta_P_Sizing: float, Ro_rel: float) -> float:
    """
    Formula 3: Q = C * N_1 * F_P * sqrt(Delta_P_Sizing / Ro_rel)

    Parameters:
        C (float): C value.
        N_1 (float): N_1 value.
        F_P (float): F_P value.
        Delta_P_Sizing (float): Delta_P_Sizing value.
        Ro_rel (float): Ro_rel value.

    Returns:
        float: Q value.
    """
    return C * N_1 * F_P * math.sqrt(Delta_P_Sizing / Ro_rel)


def equation25(Delta_P_Sizing: list, Delta_P: list, Delta_P_Chocked: list, Delta_P_chocked: list) -> list:
    """
    Formula 25: Delta_P_Sizing = Delta_P if Delta_P < Delta_P_Chocked else Delta_P_chocked

    Parameters:
        Delta_P_Sizing (list): Empty list to store Delta_P_Sizing values.
        Delta_P (list): Delta_P values.
        Delta_P_Chocked (list): Delta_P_Chocked values.
        Delta_P_chocked (list): Delta_P_chocked values.

    Returns:
        list: Delta_P_Sizing values.
    """
    for i in range(3):
        if Delta_P[i] < Delta_P_Chocked[i]:
            Delta_P_Sizing[i] = Delta_P[i]
        else:
            Delta_P_Sizing[i] = Delta_P_chocked[i]

    return Delta_P_Sizing


def equation4(Delta_P_Chocked: list, F_LP: list, F_P: list, P_1: list, F_F: list, p_v: list, F_L: list) -> list:
    """
    Formula 4: Delta_P_Chocked = ((F_LP / F_P) ^ 2) * (P_1 - F_F * p_v)

    Parameters:
        Delta_P_Chocked (list): Empty list to store Delta_P_Chocked values.
        F_LP (list): F_LP values.
        F_P (list): F_P values.
        P_1 (list): P_1 values.
        F_F (list): F_F values.
        p_v (list): p_v values.
        F_L (list): F_L values.

    Returns:
        list: Delta_P_Chocked values.
    """
    for i in range(3):
        if F_LP[i] == F_P[i]:
            Delta_P_Chocked[i] = math.pow(F_L[i], 2) * (P_1[i] - F_F[i] * p_v[i])
        else:
            Delta_P_Chocked[i] = math.pow(F_LP[i] / F_P[i], 2) * (P_1[i] - F_F[i] * p_v[i])

    return Delta_P_Chocked


def equation5_version1(F_F: list, P_v: list, P_c: list) -> list:
    """
    Formula 5: F_F = 0.96 - 0.28 * sqrt(P_v / P_c)

    Parameters:
        F_F (list): Empty list to store F_F values.
        P_v (list): P_v values.
        P_c (list): P_c values.

    Returns:
        list: F_F values.
    """
    for i in range(3):
        F_F[i] = 0.96 - 0.28 * math.sqrt(P_v[i] / P_c[i])

    return F_F


def equation5_version2(W: list, C: list, N_6: float, F_P: list, Y: list, x_sizing: list, P_1: list,
                       Ro_1: float) -> list:
    """
    Formula 5: W = C * N_6 * F_P * Y * sqrt(x_sizing * P_1 * Ro_1)

    Parameters:
        W (list): Empty list to store W values.
        C (list): C values.
        N_6 (float): N_6 value.
        F_P (list): F_P values.
        Y (list): Y values.
        x_sizing (list): x_sizing values.
        P_1 (list): P_1 values.
        Ro_1 (float): Ro_1 value.

    Returns:
        list: W values.
    """
    for i in range(3):
        W[i] = C[i] * N_6 * F_P[i] * Y[i] * math.sqrt(x_sizing[i] * P_1[i] * Ro_1)

    return W


def equation6(W: list, C: list, N_8: float, F_P: list, Y: list, x_sizing: list, P_1: list, Ro_1: float, M: list,
              T_1: list, Z_1: list) -> list:
    """
    Formula 6: W = C * N_8 * F_P * P_1 * Y * sqrt((x_sizing * M) / (T_1 * Z_1))

    Parameters:
        W (list): Empty list to store W values.
        C (list): C values.
        N_8 (float): N_8 value.
        F_P (list): F_P values.
        Y (list): Y values.
        x_sizing (list): x_sizing values.
        P_1 (list): P_1 values.
        Ro_1 (float): Ro_1 value.
        M (list): M values.
        T_1 (list): T_1 values.
        Z_1 (list): Z_1 values.

    Returns:
        list: W values.
    """
    for i in range(3):
        W[i] = C[i] * N_8 * F_P[i] * P_1[i] * Y[i] * math.sqrt(x_sizing[i] * M[i]) / (T_1[i] * Z_1[i])

    return W


def equation26(x_sizing: list, x: list, x_chocked: list) -> list:
    """
    Formula 26: x_sizing = x if x < x_chocked else x_chocked

    Parameters:
        x_sizing (list): Empty list to store x_sizing values.
        x (list): x values.
        x_chocked (list): x_chocked values.

    Returns:
        list: x_sizing values.
    """
    for i in range(3):
        x_sizing[i] = x[i] if x[i] < x_chocked[i] else x_chocked[i]

    return x_sizing


def equation27(x: list, Delta_P: list, P_1: list) -> list:
    """
    Formula 27: x = Delta_P / P_1

    Parameters:
        x (list): Empty list to store x values.
        Delta_P (list): Delta_P values.
        P_1 (list): P_1 values.

    Returns:
        list: x values.
    """
    for i in range(3):
        x[i] = Delta_P[i] / P_1[i]

    return x


def equation7_version1(x_chocked: list, F_upsilon: list, x_TP: list) -> list:
    """
    Formula 7: x_chocked = F_upsilon * x_TP

    Parameters:
        x_chocked (list): Empty list to store x_chocked values.
        F_upsilon (list): F_upsilon values.
        x_TP (list): x_TP values.

    Returns:
        list: x_chocked values.
    """
    for i in range(3):
        x_chocked[i] = F_upsilon[i] * x_TP[i]

    return x_chocked


def equation7_version2(F_upsilon: list, upsilon: list) -> list:
    """
    Heuristic Formula 7: F_upsilon = upsilon / 1.4

    Parameters:
        F_upsilon (list): Empty list to store F_upsilon values.
        upsilon (list): upsilon values.

    Returns:
        list: F_upsilon values.
    """
    for i in range(3):
        F_upsilon[i] = upsilon[i] / 1.4

    return F_upsilon


def equation8(Y: list, x_sizing: list, x_chocked: list) -> list:
    """
    Formula 8: Y = 1 - (x_sizing / (3 * x_chocked))

    Parameters:
        Y (list): Empty list to store Y values.
        x_sizing (list): x_sizing values.
        x_chocked (list): x_chocked values.

    Returns:
        list: Y values.
    """
    for i in range(3):
        Y[i] = 1 - (x_sizing[i] / (3 * x_chocked[i]))

    return Y


def equation9(P_1: list, P_c: list, P_r: list) -> list:
    """
    Formula 9: P_r = P_1 / P_c

    Parameters:
        P_1 (list): P_1 values.
        P_c (list): P_c values.
        P_r (list): Empty list to store P_r values.

    Returns:
        list: P_r values.
    """
    for i in range(3):
        P_r[i] = P_1[i] / P_c[i]

    return P_r


def equation10(T_r: list, T_1: list, T_c: list) -> list:
    """
    Formula 10: T_r = T_1 / T_c

    Parameters:
        T_r (list): Empty list to store T_r values.
        T_1 (list): T_1 values.
        T_c (list): T_c values.

    Returns:
        list: T_r values.
    """
    for i in range(3):
        T_r[i] = T_1[i] / T_c[i]

    return T_r


def equation11(F_P: list, Zeta: list, N_2: float, C: list, d: list) -> list:
    """
    Formula 11: F_P = 1 / sqrt(1 + (sum(Zeta) / N_2) * ((C / (d^2))^2))

    Parameters:
        F_P (list): Empty list to store F_P values.
        Zeta (list): Zeta values.
        N_2 (float): N_2 value.
        C (list): C values.
        d (list): d values.

    Returns:
        list: F_P values.
    """
    for i in range(3):
        F_P[i] = 1 / math.sqrt(1 + (sum(Zeta) / N_2)) * (C[i] / (d[i] ** 2)) ** 2

    return F_P


def equation12_version1(Zeta_1: list, Zeta_2: list, Zeta_B1: list, Zeta_B2: list, Zeta: list) -> list:
    """
    Formula 12: sum(Zeta) = Zeta_1 + Zeta_2 - Zeta_B1 - Zeta_B2

    Parameters:
        Zeta_1 (list): Zeta_1 values.
        Zeta_2 (list): Zeta_2 values.
        Zeta_B1 (list): Zeta_B1 values.
        Zeta_B2 (list): Zeta_B2 values.
        Zeta (list): Empty list to store sum(Zeta) values.

    Returns:
        list: sum(Zeta) values.
    """
    for i in range(3):
        Zeta[i] = Zeta_1[i] + Zeta_2[i] - Zeta_B1[i] - Zeta_B2[i]

    return Zeta


def equation12_version2(d: list, D: list, Zeta_B: list) -> list:
    """
    Formula 12: Zeta_B = 1 - (d / D)^2

    Parameters:
        d (list): d values.
        D (list): D values.
        Zeta_B (list): Empty list to store Zeta_B values.

    Returns:
        list: Zeta_B values.
    """
    for i in range(3):
        Zeta_B[i] = 1 - (d[i] / D[i]) ** 2

    return Zeta_B


def equation13(d: list, D_1: list, Zeta_1: list) -> list:
    """
    Formula 13: Zeta_1 = 0.5 * (1 - (d / D_1)^2)^2

    Parameters:
        d (list): d values.
        D_1 (list): D_1 values.
        Zeta_1 (list): Empty list to store Zeta_1 values.

    Returns:
        list: Zeta_1 values.
    """
    for i in range(3):
        Zeta_1[i] = 0.5 * (1 - (d[i] / D_1[i]) ** 2) ** 2

    return Zeta_1


def equation14(d: list, D_2: list, Zeta_2: list) -> list:
    """
    Formula 14: Zeta_2 = 1 * (1 - (d / D_2)^2)^2

    Parameters:
        d (list): d values.
        D_2 (list): D_2 values.
        Zeta_2 (list): Empty list to store Zeta_2 values.

    Returns:
        list: Zeta_2 values.
    """
    for i in range(3):
        Zeta_2[i] = 1 * (1 - (d[i] / D_2[i]) ** 2) ** 2

    return Zeta_2


def equation15_version2(d: list, D: list, Zeta_1: list, Zeta_2: list, SumZeta: list) -> list:
    """
    Formula 15: Zeta_1 + Zeta_2 = 1.5 * (1 - (d / D)^2)^2

    Parameters:
        d (list): d values.
        D (list): D values.
        Zeta_1 (list): Zeta_1 values.
        Zeta_2 (list): Zeta_2 values.
        SumZeta (list): Empty list to store Zeta_1 + Zeta_2 values.

    Returns:
        list: Zeta_1 + Zeta_2 values.
    """
    for i in range(3):
        SumZeta[i] = Zeta_1[i] + Zeta_2[i] + 1.5 * (1 - (d[i] / D[i]) ** 2) ** 2

    return SumZeta


def equation16(F_LP: list, F_L: list, N_2: float, Zeta_1: list, C: list, d: list) -> list:
    """
    Formula 16: F_LP = F_L / sqrt(1 + (F_L^2 / N_2) * (Zeta_1) * (C / d^2)^2)

    Parameters:
        F_LP (list): Empty list to store F_LP values.
        F_L (list): F_L values.
        N_2 (float): N_2 value.
        Zeta_1 (list): Zeta_1 values.
        C (list): C values.
        d (list): d values.

    Returns:
        list: F_LP values.
    """
    for i in range(3):
        F_LP[i] = F_L[i] / (1 + (F_L[i] ** 2 / N_2) * Zeta_1[i] * (C[i] / d[i] ** 2) ** 2) ** 0.5

    return F_LP


def equation17(x_TP: list, x_T: list, F_p: list, Zeta_i: list, C: list, d: list, N_5: float) -> list:
    """
    Formula 17: x_TP = (x_T / F_p^2) / (1 + (x_T * Zeta_i / N_5) * (C / d^2)^2)

    Parameters:
        x_TP (list): Empty list to store x_TP values.
        x_T (list): x_T values.
        F_p (list): F_p values.
        Zeta_i (list): Zeta_i values.
        C (list): C values.
        d (list): d values.
        N_5 (float): N_5 value.

    Returns:
        list: x_TP values.
    """
    for i in range(3):
        x_TP[i] = x_T[i] / (F_p[i] ** 2) / (1 + (x_T[i] * Zeta_i[i] / N_5) * (C[i] / d[i] ** 2) ** 2)

    return x_TP


def equation18(Re_v: list, F_d: list, Q: list, F_L: list, C: list, d: list, N_4: float, N_2: float,
               viscosity: list) -> list:
    """
    Formula 18: Re_v = (N_4 * F_d * Q * (F_L^2 * C^2 / (N_2 * d^4) + 1)^(1/4)) / (viscosity * sqrt(C * F_L))

    Parameters:
        Re_v (list): Empty list to store Re_v values.
        F_d (list): F_d values.
        Q (list): Q values.
        F_L (list): F_L values.
        C (list): C values.
        d (list): d values.
        N_4 (float): N_4 value.
        N_2 (float): N_2 value.
        viscosity (list): viscosity values.

    Returns:
        list: Re_v values.
    """
    for i in range(3):
        Re_v[i] = N_4 * F_d[i] * Q[i] * (F_L[i] ** 2 * C[i] ** 2 / (N_2 * d[i] ** 4) + 1) ** 0.25 / (
                viscosity[i] * (C[i] * F_L[i]) ** 0.5)

    return Re_v


def equation19(F_R: list, Delta_P_Actual: list, Q: list, Ro_Rel: list, C: list, N_1: float) -> list:
    """
    Formula 19: Q = C * N_1 * F_R * sqrt(Delta_P_Actual / Ro_Rel)

    Parameters:
        F_R (list): F_R values.
        Delta_P_Actual (list): Delta_P_Actual values.
        Q (list): Empty list to store Q values.
        Ro_Rel (list): Ro_Rel values.
        C (list): C values.
        N_1 (float): N_1 value.

    Returns:
        list: Q values.
    """
    for i in range(3):
        Q[i] = C[i] * N_1 * F_R[i] * (Delta_P_Actual[i] / Ro_Rel[i]) ** 0.5

    return Q


def equation20(F_R: list, Delta_P: list, W: list, C: list, N_27: float, P_1: list, P_2: list, M: list, T_1: list,
               Y: list) -> list:
    """
    Formula 20: W = C * N_27 * F_R * Y * sqrt(Delta_P * (P_1 + P_2) * M / T_1)

    Parameters:
        F_R (list): F_R values.
        Delta_P (list): Delta_P values.
        W (list): Empty list to store W values.
        C (list): C values.
        N_27 (float): N_27 value.
        P_1 (list): P_1 values.
        P_2 (list): P_2 values.
        M (list): M values.
        T_1 (list): T_1 values.
        Y (list): Y values.

    Returns:
        list: W values.
    """
    for i in range(3):
        W[i] = C[i] * N_27 * F_R[i] * Y[i] * (Delta_P[i] * (P_1[i] + P_2[i]) * M[i] / T_1[i]) ** 0.5

    return W


def equation21(F_R: list, Delta_P: list, Q_s: list, C: list, N_22: float, P_1: list, P_2: list, M: list, T_1: list,
               Y: list) -> list:
    """
    Formula 21: Q_s = C * N_22 * F_R * Y * sqrt(Delta_P * (P_1 + P_2) / (M * T_1))

    Parameters:
        F_R (list): F_R values.
        Delta_P (list): Delta_P values.
        Q_s (list): Empty list to store Q_s values.
        C (list): C values.
        N_22 (float): N_22 value.
        P_1 (list): P_1 values.
        P_2 (list): P_2 values.
        M (list): M values.
        T_1 (list): T_1 values.
        Y (list): Y values.

    Returns:
        list: Q_s values.
    """
    for i in range(3):
        Q_s[i] = C[i] * N_22 * F_R[i] * Y[i] * (Delta_P[i] * (P_1[i] + P_2[i]) / (M[i] * T_1[i])) ** 0.5

    return Q_s


def equation22(Y: list, Re_v: list, x_sizing: list, x_chocked: list, x: list) -> list:
    """
    Formula 22: Y = ((Re_v - 1000) / 9000) * (1 - x_sizing / (3 * x_chocked)) - sqrt(1 - x / 2) + sqrt(1 - x / 2) if 1000 <= Re_v < 10000
                Y = sqrt(1 - x / 2) otherwise

    Parameters:
        Y (list): Empty list to store Y values.
        Re_v (list): Re_v values.
        x_sizing (list): x_sizing values.
        x_chocked (list): x_chocked values.
        x (list): x values.

    Returns:
        list: Y values.
    """
    for i in range(3):
        if 1000 <= Re_v[i] < 10000:
            Y[i] = ((Re_v[i] - 1000) / 9000) * (1 - x_sizing[i] / (3 * x_chocked[i])) - (1 - x[i] / 2) ** 0.5 + (
                    1 - x[i] / 2) ** 0.5
        else:
            Y[i] = (1 - x[i] / 2) ** 0.5

    return Y


def equation23(F_R: list, Re_v: list, F_L: list, n: float) -> list:
    """
    Formula 23: F_R = min(0.026 * sqrt(n * Re_v) / F_L, 1)

    Parameters:
        F_R (list): Empty list to store F_R values.
        Re_v (list): Re_v values.
        F_L (list): F_L values.
        n (float): n value.

    Returns:
        list: F_R values.
    """
    for i in range(3):
        F_R[i] = min(0.026 * (n * Re_v[i]) ** 0.5 / F_L[i], 1)

    return F_R


def equation24(F_R: list, Re_v: list, F_L: list, C_rated: list, C: list, d: list, N_2: float, N_18: float,
               N_32: float) -> list:
    """
    Formula 24: F_R = min(1 + (0.33 * (F_L ** 0.5) / (n ** 0.25) * log(Re_v / 10000)), 0.026 * sqrt(n * Re_v) / F_L, 1)
                if C_rated / (d ** 2 * N_18) >= 0.0016, n = N_2 / (C / d ** 2) ** 2
                if C_rated / (d ** 2 * N_18) < 0.0016, n = (1 + N_32 * (C / d) ** (2 / 3)) ^ (2 / 3)

    Parameters:
        F_R (list): Empty list to store F_R values.
        Re_v (list): Re_v values.
        F_L (list): F_L values.
        C_rated (list): C_rated values.
        C (list): C values.
        d (list): d values.
        N_2 (float): N_2 value.
        N_18 (float): N_18 value.
        N_32 (float): N_32 value.

    Returns:
        list: F_R values.
    """
    for i in range(3):
        if C_rated[i] / (d[i] ** 2 * N_18) >= 0.0016:
            n = (N_2 / (C[i] / d[i]) ** 2) ** 2
        else:
            n = (1 + N_32 * (C[i] / d[i]) ** (2 / 3)) ** (2 / 3)
        F_R[i] = min(1 + (0.33 * (F_L[i] ** 0.5) / (n ** 0.25) * math.log(Re_v[i] / 10000)),
                     0.026 * (n * Re_v[i]) ** 0.5 / F_L[i], 1)

    return F_R
