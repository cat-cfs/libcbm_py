import pyarrow as pa
import numpy as np
import pandas as pd
from typing import Any
from typing import Union
from typing import Callable
import ctypes

SeriesInitType = Union[str, int, float, list, np.ndarray, pd.Series, pa.Array]


class Series:
    """
    Series is a wrapper for one of several underlying storage types which
    presents a limited interface for internal usage by libcbm.
    """

    def __init__(
        self,
        name: str,
        init: SeriesInitType,
        type: str = None,
    ):
        self._name = name

    def name(self) -> str:
        return self._name

    def filter(self, arg: "Series") -> "Series":
        """
        Return a new series of the elements
        corresponding to the true values in the specified arg
        """
        raise NotImplementedError()

    def take(self, indices: "Series") -> "Series":
        """return the elements of this series at the specified indices
        (returns a copy)"""
        raise NotImplementedError()

    def assign(self, indices: "Series", value: Any):
        raise NotImplementedError()

    def assign_all(self, value: Any):
        """
        set all values in this series to the specified value
        """

    def map(self, arg: Union[dict, Callable[[int, Any], Any]]) -> "Series":
        raise NotImplementedError()

    def at(self, idx: int) -> Any:
        """Gets the value at the specified sequential index"""
        raise NotImplementedError()

    def any(self) -> bool:
        """
        return True if at least one value in this series is
        non-zero
        """
        raise NotImplementedError()

    def unique(self) -> "Series":
        raise NotImplementedError()

    def to_numpy(self) -> np.ndarray:
        raise NotImplementedError()

    def to_numpy_ptr(self) -> ctypes.pointer:
        raise NotImplementedError()

    def less(self, other: "Series") -> "Series":
        """
        returns a boolean series with:
            True - where this series is less than the other series
            False - where this series is greater than or equal to the other
                series
        """
        raise NotImplementedError()

    def sum(self) -> Union[int, float]:
        raise NotImplementedError()

    def __mul__(self, other: Union[int, float, "Series"]) -> "Series":
        raise NotImplementedError()

    def __rmul__(self, other: Union[int, float, "Series"]) -> "Series":
        raise NotImplementedError()

    def __add__(self, other: Union[int, float, "Series"]) -> "Series":
        raise NotImplementedError()

    def __radd__(self, other: Union[int, float, "Series"]) -> "Series":
        raise NotImplementedError()

    def __gt__(self, other: Union[int, float, "Series"]) -> "Series":
        raise NotImplementedError()

    def __lt__(self, other: Union[int, float, "Series"]) -> "Series":
        raise NotImplementedError()

    def __eq__(sefl, other: Union[int, float, "Series"]) -> "Series":
        raise NotImplementedError()


class NullSeries(Series):
    """represents a series with no elements"""

    def __init__(self, name):
        super().__init__(name, None, None)

    def to_numpy(self) -> np.ndarray:
        return None
