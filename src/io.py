import pandas as pd


def get_all_sheets_names(excel_path: str) -> [str]:
    """
    Get all sheets names from Excel file
    :param excel_path: path to Excel file
    :return: list of sheets names
    """
    excel_file = pd.ExcelFile(excel_path)
    return excel_file.sheet_names


def is_valid_path(path: str, extension: str) -> bool:
    """
    Check if path is valid
    :param path: path to file
    :param extension: file extension
    :return: True if path is valid, False otherwise
    """
    return path.endswith(extension)


def read_sheet_by_index(excel_path: str, sheet_index: int, skiprows=None) -> pd.DataFrame:
    """
    Read sheet by index
    :param excel_path: path to excel file
    :param sheet_index: index of sheet
    :param skiprows: number of rows to skip
    :return: dataframe of sheet
    """
    sheets = get_all_sheets_names(excel_path)
    df = pd.read_excel(excel_path, sheet_name=sheets[sheet_index], skiprows=skiprows)
    df.columns = df.columns.str.strip()
    return df


def read_sheet_by_name(excel_path: str, sheet_name: str, skiprows=None) -> pd.DataFrame:
    """
    Read sheet by name
    :param excel_path: path to excel file
    :param sheet_name: name of sheet
    :param skiprows: number of rows to skip
    :return: dataframe of sheet
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=skiprows)
    df.columns = df.columns.str.strip()
    return df
