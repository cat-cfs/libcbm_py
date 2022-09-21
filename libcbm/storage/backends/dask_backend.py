from typing import Any
from typing import Union
import ctypes
import numpy as np
import pandas as pd
from libcbm.storage.dataframe import DataFrame
from libcbm.storage.series import Series
from libcbm.storage.backends import BackendType
from dask import array as da


class DaskDataFrameBackend(DataFrame):

    def __init__(self, data: dict[str, da.Array]):
        self._data = data

    def __getitem__(self, col_name: str) -> Series:
        pass

    def filter(self, arg: Series) -> "DataFrame":
        pass

    def take(self, indices: Series) -> "DataFrame":
        pass

    def at(self, index: int) -> dict:
        pass

    @property
    def n_rows(self) -> int:
        pass

    @property
    def n_cols(self) -> int:
        pass

    @property
    def columns(self) -> list[str]:
        pass

    @property
    def backend_type(self) -> BackendType:
        return BackendType.dask

    def copy(self) -> "DataFrame":
        pass

    def multiply(self, series: Series) -> "DataFrame":
        pass

    def add_column(self, series: Series, index: int) -> None:
        pass

    def to_numpy(self, make_c_contiguous=True) -> np.ndarray:
        pass

    def to_pandas(self) -> pd.DataFrame:
        pass

    def zero(self):
        pass

    def map(self, arg: dict) -> "DataFrame":
        pass

    def evaluate_filter(self, expression: str) -> Series:
        pass

    def sort_values(self, by: str, ascending: bool = True) -> "DataFrame":
        pass


class DaskSeriesBackend(Series):

    def __init__(self, name: str, data: da.Array):
        self._name = name
        self._data = data

    @property  # pragma: no cover
    def name(self) -> str:
        pass

    @name.setter  # pragma: no cover
    def name(self, value: str) -> str:
        pass

    def copy(self) -> "Series":
        pass

    def filter(self, arg: "Series") -> "Series":
        pass

    def take(self, indices: "Series") -> "Series":
        pass

    def as_type(self, type_name: str) -> "Series":
        pass

    def assign(
        self,
        value: Union["Series", Any],
        indices: "Series" = None,
        allow_type_change=False,
    ):
        pass

    def map(self, arg: dict) -> "Series":

        pass

    def at(self, idx: int) -> Any:
        pass

    def any(self) -> bool:
        pass

    def all(self) -> bool:
        pass

    def unique(self) -> "Series":
        pass

    def to_numpy(self) -> np.ndarray:
        pass

    def to_list(self) -> list:
        pass

    def to_numpy_ptr(self) -> ctypes.pointer:
        pass

    @property  # pragma: no cover
    def data(self) -> Union[np.ndarray, pd.Series]:
        pass

    def sum(self) -> Union[int, float]:
        pass

    def cumsum(self) -> "Series":
        pass

    def max(self) -> Union[int, float]:
        pass

    def min(self) -> Union[int, float]:
        pass

    @property  # pragma: no cover
    def length(self) -> int:
        pass

    @property  # pragma: no cover
    def backend_type(self) -> BackendType:
        pass

    def __mul__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __rmul__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __truediv__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __rtruediv__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __add__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __radd__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __sub__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __rsub__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __ge__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __gt__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __le__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __lt__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __eq__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __ne__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __and__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __or__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __rand__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __ror__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    def __invert__(self) -> "Series":
        pass


def concat_series(series: list[DaskSeriesBackend]) -> DaskSeriesBackend:
    return DaskSeriesBackend(
        None, np.concatenate([s._get_data() for s in series])
    )


def logical_and(
    s1: DaskSeriesBackend, s2: DaskSeriesBackend
) -> DaskSeriesBackend:
    return DaskSeriesBackend(
        None, np.logical_and(s1._get_data(), s2._get_data())
    )


def logical_not(series: DaskSeriesBackend) -> DaskSeriesBackend:
    return DaskSeriesBackend(None, np.logical_not(series._get_data()))


def logical_or(
    s1: DaskSeriesBackend, s2: DaskSeriesBackend
) -> DaskSeriesBackend:
    return DaskSeriesBackend(
        None, np.logical_or(s1._get_data(), s2._get_data())
    )


def make_boolean_series(init: bool, size: int) -> DaskSeriesBackend:
    return DaskSeriesBackend(
        None, np.full(shape=size, fill_value=init, dtype="bool")
    )


def is_null(series: DaskSeriesBackend) -> DaskSeriesBackend:
    return DaskSeriesBackend(None, pd.isnull(series._get_data()))


def indices_nonzero(series: DaskSeriesBackend) -> DaskSeriesBackend:
    return DaskSeriesBackend(None, np.nonzero(series._get_data())[0])


def numeric_dataframe(
    cols: list[str],
    nrows: int,
    init: float = 0.0,
) -> DaskDataFrameBackend:
    pass


def from_series_list(
    series_list: list[DaskSeriesBackend],
) -> DaskDataFrameBackend:
    pass


def from_series_dict(
    data: dict[str, DaskSeriesBackend],
) -> DataFrame:
    pass


def allocate(name: str, len: int, init: Any, dtype: str) -> DaskSeriesBackend:
    pass

def range(
    name: str,
    start: int,
    stop: int,
    step: int,
    dtype: str,
) -> Series:
    pass
