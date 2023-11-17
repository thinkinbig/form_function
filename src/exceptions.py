class FormatError(Exception):
    """
    通用参数格式错误
    """
    pass


class SteamFormatError(Exception):
    """
    蒸汽表格格式错误
    """
    pass


class EmptyException(Exception):
    """
    不在范围内的空值 或者 空值错误
    """
    pass


class QueryBeforeCalculationError(Exception):
    """
    计算前查询错误
    """
    pass


class MediumStatusError(Exception):
    """
    介质状态不支持错误
    """
    pass


class ExtractError(Exception):
    """
    提取错误
    """
    pass


class NotAttachedException(Exception):
    """
    未添加错误
    """
    pass


class UnitConversionError(Exception):
    """
    单位转换错误
    """
    pass


class LeakingLevelError(Exception):
    """
    泄漏级别错误
    """
    pass


class UploadFileError(Exception):
    pass
