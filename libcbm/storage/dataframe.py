import numpy as np
import pandas as pd

from typing import Union
from typing import Callable
from libcbm.storage.backends import BackendType
from libcbm.storage.series import Series
from typing import Any
from libcbm.storage.backends import factory


class DataFrame:
    """
    DataFrame is a wrapper for one of several underlying storage types which
    presents a limited interface for internal usage by libcbm
    """

    def __init__(
        self,
        data: Union[dict[str, Series], list[Series], pd.DataFrame],
        nrows: int = None,
        back_end: BackendType = None,
    ):

        self._data = data
        self._back_end = factory.get_backend(back_end)

    def __getitem__(self, col_name: str) -> Series:
        return self._back_end.getitem(self._data, col_name)

    def filter(self, arg: Series) -> "DataFrame":
        return self._back_end.filter(self._data, arg)

    def take(self, indices: Series) -> "DataFrame":
        return self._back_end.take(self._data, indices)

    def at(self, index: int) -> dict:
        """
        get the row at the specified 0 based sequential index as a row
        dictionary
        """
        return self._back_end.at(self._data, index)

    def assign(self, col_name: str, value: Any, indices: Series = None):
        self._back_end.assign(self._data, col_name, value, indices)

    @property
    def n_rows(self) -> int:
        return self._back_end.n_rows(self._data)

    @property
    def n_cols(self) -> int:
        return self._back_end.n_cols(self._data)

    @property
    def columns(self) -> list[str]:
        return self._back_end.columns(self._data)

    @property
    def backend_type(self) -> BackendType:
        return self._back_end.backend_type()

    def copy(self) -> "DataFrame":
        """produce a new in-memory copy of this dataframe"""
        return self._back_end.copy(self._data)

    def multiply(self, series: Series) -> "DataFrame":
        """
        Multiply this dataframe elementwise by the specified series along the
        row axis. An error is raised if the series length is not the same as
        the number of rows in this dataframe.  Returns new DataFrame
        """
        return self._back_end.multiply(self._data, series)

    def add_column(self, series: Series, index: int) -> None:
        self._back_end.add_column(self._data, series, index)

    def to_c_contiguous_numpy_array(self) -> np.ndarray:
        return self._back_end.to_c_contiguous_numpy_array(self._data)

    def to_pandas(self) -> pd.DataFrame:
        return self._back_end.to_pandas(self._data)

    def zero(self):
        """
        Set all values in this dataframe to zero
        """
        self._back_end.zero(self._data)

    def map(self, arg: Union[dict, Callable]) -> "DataFrame":
        """Apply the specified mapping arg on every element of this dataframe
        to project a new dataframe with updated values. The results has the
        same number of rows, columns and same column names
        """
        return self._back_end.map(self._data, arg)


def numeric_dataframe(
    cols: list[str],
    nrows: int,
    init: float = 0.0,
    back_end: BackendType = BackendType.numpy,
) -> DataFrame:
    raise NotImplementedError()


def from_pandas(df: pd.DataFrame) -> DataFrame:
    return DataFrame(df, back_end=BackendType.pandas)
