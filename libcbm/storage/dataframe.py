import numpy as np
import pandas as pd

from typing import Union
from typing import Callable
from libcbm.storage.backends import BackendType
from libcbm.storage.series import Series
from typing import Any


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
        raise NotImplementedError()

    def __getitem__(self, col_name: str) -> Series:
        raise NotImplementedError()

    def filter(self, arg) -> "DataFrame":
        pass

    def take(self, indices) -> "DataFrame":
        pass

    def at(self, index) -> dict:
        """
        get the row at the specified 0 based sequential index as a row
        dictionary
        """
        pass

    def assign(self, col_name: str, value: Any, indices: Series = None):
        pass

    @property
    def n_rows(self) -> int:
        raise NotImplementedError()

    @property
    def n_cols(self) -> int:
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

    def to_c_contiguous_numpy_array(self) -> np.ndarray:
        raise NotImplementedError()

    def to_pandas(self) -> pd.DataFrame:
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
    init: float = 0.0,
    back_end: BackendType = BackendType.numpy,
) -> DataFrame:
    raise NotImplementedError()


def from_pandas(df: pd.DataFrame) -> DataFrame:
    return DataFrame(df, back_end=BackendType.pandas)
