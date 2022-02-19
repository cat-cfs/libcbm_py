from libcbm.storage.dataframe import DataFrame
from libcbm.storage.dataframe import Series


def concat_data_frame(dfs: list[DataFrame]):
    raise NotImplementedError()


def concat_series(series: list[Series]):
    raise NotImplementedError()


def logical_and(s1: Series, s2: Series) -> Series:
    raise NotImplementedError()


def logical_not(series: Series) -> Series:
    raise NotImplementedError()


def logical_or(series: Series) -> Series:
    raise NotImplementedError()


def make_boolean_series(init: bool, size: int) -> Series:
    raise NotImplementedError()
