import logging

from src.exceptions import EmptyException, FormatError, ExtractError, QueryBeforeCalculationError, UnitConversionError


def ignore_value(func, post_process=None):
    """
    Decorator to ignore empty value
    注意：这个装饰器只能用于函数，不能用于类
    并且只能作用在需要忽略空值的函数上， 一旦忽略空值，就会退出函数
    Parameters
    ----------
    func : function
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except EmptyException as e:
            logging.debug(e)
            pass
        except FormatError as e:
            logging.debug(e)
            pass
        except QueryBeforeCalculationError as e:
            logging.debug(e)
            pass
        except UnitConversionError as e:
            logging.debug(e)
            pass
        except ValueError as e:
            logging.debug(e)
            pass
        except Exception as e:
            logging.error(e)
            import traceback
            traceback.print_exception(e)
            pass

    return wrapper


def in_to_mm(string: str) -> float:
    """
    将英寸转换为毫米
    Parameters
    ----------
    string : str 传入的字符串 可能是英寸，可能是毫米

    Returns
    -------
    float: 返回毫米

    """
    is_inch = False
    if "*" in string:
        # 如果有缩径，即存在xx*xx的情况，则取*后的数值
        string = string.split("*")[1]
    if string.startswith('DN'):
        # 如果是DN，则转换为毫米
        string = string[2:]
    if "''" in string or "“" in string or '"' in string or "in" in string:
        # 如果是英寸，则转换为毫米
        string = string.replace("''", "").replace("“", "").replace('"', "").replace("in", "")
        is_inch = True
    if "mm" in string or "MM" in string:
        # 如果是毫米，则直接返回
        string = string.replace("mm", "").replace("MM", "")
    try:  # 如果是数字，则直接返回
        result = float(string)
        if is_inch:
            logging.debug(f"inch: {result}")
            result *= 25.4
        return result
    except ValueError:
        raise FormatError("格式错误")


def is_inch(string: str) -> bool:
    return "''" in string or "“" in string or '"' in string or "in" in string


def replace_inch(string: str) -> str:
    assert is_inch(string)
    return string.replace("''", "").replace("“", "").replace('"', "").replace("in", "")


def ignore_unit(func):
    """
    Decorator to ignore unit
    注意：这个装饰器只能用于函数，不能用于类
    并且只能作用在需要忽略单位的函数上
    NOTE!!!: 如果需要忽略空值，要把这个装饰器放在ignore_value装饰器的外面
    Parameters
    ----------
    func : function
    """

    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        if isinstance(value, str):
            value = value.split(" ")[0] \
                .lower() \
                .replace("bar", "") \
                .replace("mbar", "") \
                .replace("g/mol", "") \
                .replace("m/s", "") \
                .replace("mm", "") \
                .replace("kg/m3", "") \
                .replace("m3/h", "")
            if value == "":
                raise EmptyException(f"只有单位，没有数值")
            return float(value)
        elif is_number(value):
            return value
        else:
            raise FormatError("格式错误")

    return wrapper


def is_number(string: str):
    """
    判断字符串是否为数字
    Parameters
    ----------
    string : str

    Returns
    -------
    bool
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def float_to_percent_str(my_float: float):
    return f"{int(my_float * 100)}%"


def float_rounding(my_float: float, digit: int = 2):
    return round(my_float, digit)


def is_char(string: str):
    """
    判断字符串是否为字符
    Parameters
    ----------
    string : str

    Returns
    -------
    bool
    """
    if len(string) == 1:
        return True
    else:
        return False


def special_roman_to_int(roman_numeral: str):
    roman_bias = ord('Ⅰ') - 1  # 8543

    roman_values = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000,
    }
    if is_char(roman_numeral) and roman_bias < ord(roman_numeral) < roman_bias + 20:
        return ord(roman_numeral) - roman_bias
    else:
        total = 0
        for i, c in enumerate(roman_numeral):
            if (i + 1) == len(roman_numeral) or roman_values[c] >= roman_values[roman_numeral[i + 1]]:
                total += roman_values[c]
            else:
                total -= roman_values[c]
        return total


def extract_value_before_keyword(s: str) -> tuple[int, int]:
    boundary = s.find('-')  # 找到数值的边界

    if boundary is None:
        raise ExtractError("提取错误, 没有找到边界-")

    mid = start = boundary + 1

    for i in range(boundary, len(s)):
        if s[i].lower() == 'x' or s[i] == '*':
            mid = i  # 找到数值的分割点
            break

    end = mid + 1
    while end < len(s) and s[end].isdigit():
        end += 1  # 找到数值的结束点

    if mid == start:
        # 如果没有找到分割点，则直接返回
        if not is_number(s[start:end]):
            raise ExtractError("提取错误, 数值不是数字")
        return int(s[start:end]), int(s[start:end])

    if not is_number(s[start:mid]) or not is_number(s[mid + 1:end]):
        raise ExtractError("提取错误, 数值不是数字")
    return int(s[start:mid]), int(s[mid + 1:end])


def extract_value(s: str) -> tuple[int, int]:
    logging.debug(f"extract_value: {s}")
    gctj, sj = extract_value_before_keyword(s)
    return gctj, sj


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'xlsx', 'xls', 'doc', 'docx', 'ppt', 'pptx'}
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
