import numpy as np
import pandas as pd
import pyarrow as pa
from typing import Union
from typing import Callable
from typing import Any

from libcbm.storage.backends import BackendType


SeriesInitType = Union[int, float, list, np.ndarray, pd.Series, pa.Array]


class Series:
    """
    Series is a wrapper for one of several underlying storage types which
    presents a limited interface for internal usage by libcbm.
    """

    def __init__(
        self,
        name: str,
        init: SeriesInitType,
        type: str,
    ):
        raise NotImplementedError()

    def __getitem__(self, arg: "Series") -> "Series":
        """
        Return a new series of the elements
        corresponding to the true values in the specified arg
        """
        raise NotImplementedError()

    def map(self, arg: Union[dict, Callable]) -> "Series":
        raise NotImplementedError()

    def unique(self) -> "Series":
        raise NotImplementedError()


class NullSeries(Series):
    """represents a series with no elements"""

    def __init__(self, name):
        super().__init__(name, None, None)


class DataFrame:
    """
    DataFrame is a wrapper for one of several underlying storage types which
    presents a limited interface for internal usage by libcbm
    """

    def __init__(
        self,
        data: list[Series],
        nrows: int,
        back_end: BackendType = BackendType.numpy,
    ):
        raise NotImplementedError()

    def __getitem__(self, col_name) -> Series:
        raise NotImplementedError()

    @property
    def n_rows(self) -> int:
        raise NotImplementedError()

    @property
    def columns(self) -> list[str]:
        raise NotImplementedError()

    @property
    def backend_type(self) -> BackendType:
        raise NotImplementedError()

    def copy() -> "DataFrame":
        """produce a new in-memory copy of this dataframe"""
        raise NotImplementedError()

    def multiply(self, series: Series) -> "DataFrame":
        """
        Multiply this dataframe elementwise by the specified series along the
        row axis. An error is raised if the series length is not the same as
        the number of rows in this dataframe.  Returns new DataFrame
        """
        raise NotImplementedError()

    def add_column(self, series: Series, index: int) -> None:
        raise NotImplementedError()

    def as_c_contiguous_matrix(self) -> np.ndarray:
        raise NotImplementedError()

    def zero(self):
        """
        Set all values in this dataframe to zero
        """
        raise NotImplementedError()

    def map(self, arg: Union[dict, Callable]) -> "DataFrame":
        """Apply the specified mapping arg on every element of this dataframe
        to project a new dataframe with updated values. The results has the
        same number of rows, columns and same column names
        """
        raise NotImplementedError()


def numeric_dataframe(
    cols: list[str],
    nrows: int,
    ncols: int,
    init: float = 0.0,
    back_end: BackendType = BackendType.numpy,
) -> DataFrame:
    raise NotImplementedError()
