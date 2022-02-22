import numpy as np
import pandas as pd
import pyarrow as pa
from typing import Union
from typing import Callable
from libcbm.storage.backends import BackendType


class SeriesInitType(Union[int, float, list, np.ndarray, pd.Series, pa.Array]):
    pass


class Series:
    def __init__(
        self,
        name: str,
        init: SeriesInitType,
        type: str,
    ):
        raise NotImplementedError()

    def map(self, arg: Union[dict, Callable]) -> "Series":
        raise NotImplementedError()


class DataFrame:
    def __init__(self, data):
        raise NotImplementedError()

    def __getitem__(self) -> Series:
        pass

    @property
    def columns(self) -> list[str]:
        pass

    @property
    def back_end(self) -> BackendType:
        pass


def numeric_dataframe(
    cols: list[str],
    nrows: int,
    ncols: int,
    init: float = 0.0,
    back_end: BackendType = BackendType.numpy,
) -> DataFrame:
    pass


def series_list_dataframe(
    data: list[Series],
    nrows: int,
    back_end: BackendType = BackendType.numpy,
) -> DataFrame:
    pass
