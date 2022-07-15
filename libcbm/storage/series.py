import pandas as pd
import numpy as np
from typing import Any
from typing import Union
from typing import Callable
import ctypes
from libcbm.storage.backends import BackendType
from libcbm.storage.backends import get_backend
from abc import ABC
from abc import abstractmethod


class Series(ABC):
    """
    Series is a wrapper for one of several underlying storage types which
    presents a limited interface for internal usage by libcbm.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @name.setter
    @abstractmethod
    def name(self, value) -> str:
        pass

    @abstractmethod
    def copy(self) -> "Series":
        pass

    @abstractmethod
    def filter(self, arg: "Series") -> "Series":
        """
        Return a new series of the elements
        corresponding to the true values in the specified arg
        """
        pass

    @abstractmethod
    def take(self, indices: "Series") -> "Series":
        """return the elements of this series at the specified indices
        (returns a copy)"""
        pass

    @abstractmethod
    def as_type(self, type_name: str) -> "Series":
        pass

    @abstractmethod
    def assign(
        self,
        indices: "Series",
        value: Union["Series", Any],
        allow_type_change=False,
    ):
        pass

    @abstractmethod
    def assign_all(self, value: Union["Series", Any], allow_type_change=False):
        """
        set all values in this series to the specified value
        """
        pass

    @abstractmethod
    def map(self, arg: Union[dict, Callable[[int, Any], Any]]) -> "Series":
        pass

    @abstractmethod
    def at(self, idx: int) -> Any:
        """Gets the value at the specified sequential index"""
        pass

    @abstractmethod
    def any(self) -> bool:
        """
        return True if at least one value in this series is
        non-zero
        """
        pass

    @abstractmethod
    def all(self) -> bool:
        """
        return True if all values in this series are non-zero
        """
        pass

    @abstractmethod
    def unique(self) -> "Series":
        pass

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        pass

    @abstractmethod
    def to_list(self) -> list:
        pass

    @abstractmethod
    def to_numpy_ptr(self) -> ctypes.pointer:
        pass

    @staticmethod
    def _get_operand(
        op: Union[int, float, "Series"]
    ) -> Union[int, float, np.ndarray, pd.DataFrame]:
        if isinstance(op, Series):
            return op.data
        return op

    @property
    @abstractmethod
    def data(self) -> Union[np.ndarray, pd.Series]:
        pass

    @abstractmethod
    def less(self, other: "Series") -> "Series":
        """
        returns a boolean series with:
            True - where this series is less than the other series
            False - where this series is greater than or equal to the other
                series
        """
        pass

    @abstractmethod
    def sum(self) -> Union[int, float]:
        pass

    @abstractmethod
    def cumsum(self) -> "Series":
        pass

    @abstractmethod
    def max(self) -> Union[int, float]:
        pass

    @abstractmethod
    def min(self) -> Union[int, float]:
        pass

    @property
    @abstractmethod
    def length(self) -> int:
        pass

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        pass

    @abstractmethod
    def __mul__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __rmul__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __truediv__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __rtruediv__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __add__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __radd__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __sub__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __rsub__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __ge__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __gt__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __le__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __lt__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __eq__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __ne__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __and__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __or__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __rand__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __ror__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod
    def __invert__(self) -> "Series":
        pass


class SeriesDef:
    def __init__(self, name: str, init: Any, dtype: str):
        self.name = name
        self.init = init
        self.dtype = dtype

    def make_series(self, len: int, back_end: BackendType):
        return allocate(self.name, len, self.init, self.dtype, back_end)


def allocate(
    name: str, len: int, init: Any, dtype: str, back_end: BackendType
) -> Series:
    return get_backend(back_end).allocate(name, len, init, dtype)


def range(
    name: str,
    start: int,
    stop: int,
    step: int,
    dtype: str,
    back_end: BackendType,
) -> Series:
    return get_backend(back_end).range(name, start, stop, step, dtype)


def from_pandas(series: pd.Series, name: str = None) -> Series:
    return get_backend(BackendType.pandas).PandasSeriesBackend(
        name if name else series.name, series=series
    )


def from_numpy(name: str, data: np.ndarray) -> Series:
    return get_backend(BackendType.numpy).NumpySeriesBackend(name, data)


def from_list(name: str, data: list) -> Series:
    return get_backend(BackendType.numpy).NumpySeriesBackend(
        name, np.array(data)
    )
