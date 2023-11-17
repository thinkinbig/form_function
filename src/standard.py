import numpy as np

from src.exceptions import UnitConversionError

KELVIN_CONST = 273.15
# Constants
N1 = 1
N2 = 0.0016
N4 = 0.0707
N5 = 0.0018
N6 = 31.6
N8 = 110
N9 = 2460
N17 = 0.00105
N18 = 0.865
N19 = 2.5
N22 = 1730
N27 = 77.5
N32 = 140


class PressureUnit:
    BAR = "Bar"
    PA = "Pa"
    MPA = "MPa"
    MPA_G = "MPa-g"
    MPA_A = "MPa-a"
    PSI = "Psi"
    KPA = "KPa"
    EMPTY = ""

    @staticmethod
    def convert(value: float, unit: str) -> float:
        if unit == PressureUnit.BAR:
            return value
        elif unit == PressureUnit.PA:
            return value * 0.00001
        elif unit == PressureUnit.MPA:
            return value * 10
        elif unit == PressureUnit.MPA_G:
            return value * 10 + 1.01
        elif unit == PressureUnit.MPA_A:
            return value * 10
        elif unit == PressureUnit.PSI:
            return value / 14.5038
        elif unit == PressureUnit.KPA:
            return value * 0.01
        elif unit == PressureUnit.EMPTY:
            return value
        else:
            raise UnitConversionError(f"Unknown pressure unit: {unit}")


class TemperatureUnit:
    K = "K"
    C = "℃"
    F = "F"
    EMPTY = ""

    @staticmethod
    def convert(value: float, unit: str) -> float:
        if unit == TemperatureUnit.K:
            return value
        elif unit == TemperatureUnit.C:
            return value + KELVIN_CONST
        elif unit == TemperatureUnit.F:
            return (value - 32) * 5 / 9 + KELVIN_CONST
        elif unit == TemperatureUnit.EMPTY:
            return value
        else:
            raise UnitConversionError(f"Unknown temperature unit: {unit}")


class DensityUnit:
    KG_M3 = "kg/m3"
    KG_L = "kg/L"
    T_M3 = "T/m3"
    EMPTY = ""

    @staticmethod
    def convert(value: float, unit: str) -> float:
        if unit == DensityUnit.KG_M3:
            return value
        elif unit == DensityUnit.KG_L:
            return value * 1000
        elif unit == DensityUnit.T_M3:
            return value * 1000
        elif unit == DensityUnit.EMPTY:
            return value
        else:
            raise UnitConversionError(f"Unknown density unit: {unit}")


class FlowUnit:
    M3_H = "M3/H"
    KG_H = "kg/h"
    L_H = "L/h"
    L_MIN = "L/min"
    EMPTY = ""

    @staticmethod
    def convert(value: float, unit: str) -> float:
        if unit == FlowUnit.M3_H:
            return value
        elif unit == FlowUnit.KG_H:
            return value
        elif unit == FlowUnit.L_H:
            return value * 0.001
        elif unit == FlowUnit.L_MIN:
            return value / 16.667
        elif unit == FlowUnit.EMPTY:
            return value
        else:
            raise UnitConversionError(f"Unknown flow unit: {unit}")

    @staticmethod
    def tune_flow(rho: float, value: float, unit: str) -> float:
        """
        如果液体流量单位是kg/h，则除以密度
        Parameters
        ----------
        rho
        value
        unit

        Returns
        -------

        """
        if rho == 0 or rho is None:
            raise ValueError(f"Unknown density: {rho}")
        if unit == FlowUnit.KG_H:
            return value / rho
        else:
            return value


class MolecularWeightUnit:
    G_MOL = "g/mol"
    EMPTY = ""

    @staticmethod
    def convert(value: float, unit: str) -> float:
        if unit == MolecularWeightUnit.G_MOL:
            return value
        elif unit == MolecularWeightUnit.EMPTY:
            return value
        else:
            raise UnitConversionError(f"Unknown molecular weight unit: {unit}")


STANDARD_UNIT = {
    "pressure": PressureUnit.BAR,
    "temperature": TemperatureUnit.K,
    "density": DensityUnit.KG_M3,
    "flow": FlowUnit.M3_H,
    "molecular_weight": MolecularWeightUnit.G_MOL,
}

# 标准管径
STANDARD_DN = np.array([
    10, 15, 20, 25, 32, 40, 50, 65, 80, 100,
    125, 150, 200, 250, 300, 350, 400, 450, 500,
    600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800,
    2000])


def get_standard_diameter(value: float) -> int:
    """
    Get standard diameter
    Parameters
    ----------
    value : float
        Diameter value
    Returns
    -------
    float
        Standard diameter
    """
    return STANDARD_DN[np.argmin(np.abs(STANDARD_DN - value))]
