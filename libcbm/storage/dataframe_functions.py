from libcbm.storage.dataframe import DataFrame
from libcbm.storage.dataframe import Series


def concat_data_frame(dfs: list[DataFrame]) -> DataFrame:
    raise NotImplementedError()


def concat_series(series: list[Series]) -> Series:
    raise NotImplementedError()


def logical_and(s1: Series, s2: Series) -> Series:
    raise NotImplementedError()


def logical_not(series: Series) -> Series:
    raise NotImplementedError()


def logical_or(series: Series) -> Series:
    raise NotImplementedError()


def make_boolean_series(init: bool, size: int) -> Series:
    raise NotImplementedError()


def is_null(series: Series) -> Series:
    """
    returns an Series of True where any value in the specified series
    is null, None, or NaN, otherwise False
    """
    raise NotImplementedError()


def indices_nonzero(series: Series) -> Series:
    """
    return a series of the 0 based sequential index of non-zero values in
    the specfied series
    """
    raise NotImplementedError()