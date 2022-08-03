import pandas as pd
import numpy as np
from typing import Any
from typing import Union
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

    @property  # pragma: no cover
    @abstractmethod
    def name(self) -> str:
        """get the series name"""
        pass

    @name.setter  # pragma: no cover
    @abstractmethod
    def name(self, value: str) -> str:
        """assign the series name"""
        pass

    @abstractmethod  # pragma: no cover
    def copy(self) -> "Series":
        """create a copy of this series"""
        pass

    @abstractmethod  # pragma: no cover
    def filter(self, arg: "Series") -> "Series":
        """
        Return a new series of the elements
        corresponding to the true values in the specified arg
        """
        pass

    @abstractmethod  # pragma: no cover
    def take(self, indices: "Series") -> "Series":
        """
        Return the elements of this series at the specified indices
        (returns a copy)
        """
        pass

    @abstractmethod  # pragma: no cover
    def as_type(self, type_name: str) -> "Series":
        """
        Return a copied series with all elements converted to the specified
        type.  A ValueError is raised if it is not possible to convert any
        element
        """
        pass

    @abstractmethod  # pragma: no cover
    def assign(
        self,
        value: Union["Series", Any],
        indices: "Series" = None,
        allow_type_change=False,
    ):
        """
        Assign a single value, or a Series to a subset or to the entirety of
        this series

        Args:
            value (Union[Series, Any]): assignment value
            indices (Series, optional): The indices of assignment.
                If unspecified all indicies are assigned. Defaults to None.
            allow_type_change (bool, optional): If set to true, the underlying
                storage type may be changed by this operation, and if False
                (default) any assignement that would result in a type change
                will result in a ValueError. Defaults to False.
        """
        pass

    @abstractmethod  # pragma: no cover
    def map(self, arg: dict) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def at(self, idx: int) -> Any:
        """Gets the value at the specified sequential index"""
        pass

    @abstractmethod  # pragma: no cover
    def any(self) -> bool:
        """
        return True if at least one value in this series is
        non-zero
        """
        pass

    @abstractmethod  # pragma: no cover
    def all(self) -> bool:
        """
        return True if all values in this series are non-zero
        """
        pass

    @abstractmethod  # pragma: no cover
    def unique(self) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def to_numpy(self) -> np.ndarray:
        pass

    @abstractmethod  # pragma: no cover
    def to_list(self) -> list:
        pass

    @abstractmethod  # pragma: no cover
    def to_numpy_ptr(self) -> ctypes.pointer:
        pass

    @staticmethod  # pragma: no cover
    def _get_operand(
        op: Union[int, float, "Series"]
    ) -> Union[int, float, np.ndarray, pd.Series]:
        if isinstance(op, Series):
            return op.data
        return op

    @property  # pragma: no cover
    @abstractmethod
    def data(self) -> Union[np.ndarray, pd.Series]:
        pass


    @abstractmethod  # pragma: no cover
    def sum(self) -> Union[int, float]:
        pass

    @abstractmethod  # pragma: no cover
    def cumsum(self) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def max(self) -> Union[int, float]:
        pass

    @abstractmethod  # pragma: no cover
    def min(self) -> Union[int, float]:
        pass

    @property  # pragma: no cover
    @abstractmethod
    def length(self) -> int:
        pass

    @property  # pragma: no cover
    @abstractmethod
    def backend_type(self) -> BackendType:
        pass

    @abstractmethod  # pragma: no cover
    def __mul__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __rmul__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __truediv__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __rtruediv__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __add__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __radd__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __sub__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __rsub__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __ge__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __gt__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __le__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __lt__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __eq__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __ne__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __and__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __or__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __rand__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
    def __ror__(self, other: Union[int, float, "Series"]) -> "Series":
        pass

    @abstractmethod  # pragma: no cover
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
    """
    method to allocate a numpy-backed series from a python list, intended
    primarily for unit and integration testing purposes

    Args:
        name (str): name of the series
        data (list): series data

    Returns:
        Series: a series
    """
    return get_backend(BackendType.numpy).NumpySeriesBackend(
        name, np.array(data)
    )
