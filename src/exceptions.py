class FormatError(Exception):
    pass


class SteamFormatError(Exception):
    pass


class EmptyException(Exception):
    pass


class QueryBeforeCalculationError(Exception):
    pass


class MediumStatusError(Exception):
    pass


class NotAttachedException(Exception):
    pass


class UnitConversionError(Exception):
    pass


class LeakingLevelError(Exception):
    pass
