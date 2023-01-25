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
        """get or set the series name

        Returns:
            str: series name
        """
        pass

    @name.setter  # pragma: no cover
    @abstractmethod
    def name(self, value: str) -> str:
        pass

    @abstractmethod  # pragma: no cover
    def copy(self) -> "Series":
        """create a copy of this series

        Returns:
            Series: a copy of this series
        """
        pass

    @abstractmethod  # pragma: no cover
    def filter(self, arg: "Series") -> "Series":
        """Return a new series of the elements corresponding to the true
        values in the specified arg

        Args:
            arg (Series): the boolean series to filter by

        Returns:
            Series: the filtered series
        """
        pass

    @abstractmethod  # pragma: no cover
    def take(self, indices: "Series") -> "Series":
        """Return the elements of this series at the specified indices
        (returns a copy)

        Args:
            indices (Series): the indices to take

        Returns:
            Series: the series values from this instance correpsonding to the
                provided indices
        """
        pass

    @abstractmethod
    def is_null(self) -> "Series":
        """Returns true where items in this series are None, or nan

        Returns:
            Series: boolean series indicating positions of None or nan in this
                series
        """
        pass

    @abstractmethod  # pragma: no cover
    def as_type(self, type_name: str) -> "Series":
        """Return a copied series with all elements converted to the specified
        type.  A ValueError is raised if it is not possible to convert any
        element contained in this series to the specified type

        Args:
            type_name (str): the type to convert to

        Returns:
            Series: the type converted series
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
        """Map the values in this series using the specified dictionary.
        The value of the returned series is the dictionary value corresponding
        to the dictionary key found in the input series.  If any element in
        this series is not defined in the specified dictionary, a ValueError
        is raised.

        Args:
            arg (dict): a dictionary of map values

        Returns:
            Series: the mapped series
        """
        pass

    @abstractmethod  # pragma: no cover
    def at(self, idx: int) -> Any:
        """Gets the value at the specified sequential index

        Args:
            idx (int): the index

        Returns:
            Any: The value at the index
        """
        pass

    @abstractmethod  # pragma: no cover
    def any(self) -> bool:
        """return True if at least one value in this series is
        non-zero

        Returns:
            bool: true if one or more element is non-zero, and
                otherwise false
        """
        pass

    @abstractmethod  # pragma: no cover
    def all(self) -> bool:
        """return True if all values in this series are non-zero

        Returns:
            bool: True if all elements are non-zero and otherwise false
        """
        pass

    @abstractmethod  # pragma: no cover
    def unique(self) -> "Series":
        """Get the distinct values in this series, as a new series

        Returns:
            Series: the unique set of values in this series
        """
        pass

    @abstractmethod  # pragma: no cover
    def to_numpy(self) -> np.ndarray:
        """Get the series values as a numpy array. This is a reference to the
        underlying memory where possible. The following summarize this::

          * pandas backend: a reference is returned
          * numpy backend (mixed column types): a reference is returned
          * numpy backend (2d matrix storage): a copy is returned

        Returns:
            np.ndarray: the series values as a numpy array, either as a
                reference or copy
        """
        pass

    @abstractmethod  # pragma: no cover
    def to_list(self) -> list:
        """returns a copy of the values in this series as a list

        Returns:
            list: the series as a python list
        """
        pass

    @abstractmethod  # pragma: no cover
    def to_numpy_ptr(self) -> ctypes.pointer:
        """Get a ctypes pointer to this series underlying numpy
        array.  In the case of numpy backend 2d matrix storage,
        a pointer to a copy of the series value is returned.

        Returns:
            ctypes.pointer: a ctypes pointer for the numpy array
        """
        pass

    @staticmethod  # pragma: no cover
    def _get_operand(
        op: Union[int, float, "Series"]
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        helper method to unpack series operator operands
        """
        if isinstance(op, Series):
            return op.data
        return op

    @property  # pragma: no cover
    @abstractmethod
    def data(self) -> Union[np.ndarray, pd.Series]:
        """get a reference to the underlying storage.

        Returns:
            Union[np.ndarray, pd.Series]: reference to the series storage
        """
        pass

    @abstractmethod  # pragma: no cover
    def sum(self) -> Union[int, float]:
        """get the sum of the series

        Returns:
            Union[int, float]: the series sum
        """
        pass

    @abstractmethod  # pragma: no cover
    def cumsum(self) -> "Series":
        """Compute the cumulative sums of the series.
        Return value is of equal length as this series.

        Returns:
            Series: the cumulative sum
        """
        pass

    @abstractmethod  # pragma: no cover
    def max(self) -> Union[int, float]:
        """Return the maximum value in the series

        Returns:
            Union[int, float]: the maximum value in the series
        """
        pass

    @abstractmethod  # pragma: no cover
    def min(self) -> Union[int, float]:
        """Return the minimum value in the series

        Returns:
            Union[int, float]: the minimum value in the series
        """
        pass

    @property  # pragma: no cover
    @abstractmethod
    def length(self) -> int:
        """Return the number of elements in this series

        Returns:
            int: the number of elements in the series
        """
        pass

    @property  # pragma: no cover
    @abstractmethod
    def backend_type(self) -> BackendType:
        """
        gets the BackendType of this series

        Returns:
            BackendType: the backend type of the series
        """
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
    """
    Information/Factory class to allocate initialized series for Series
    and DataFrame construction
    """

    def __init__(self, name: str, init: Any, dtype: str):
        """Initialize SeriesDef

        Args:
            name (str): the name of the series to build
            init (Any): the initialization value for all elements in the built
                series
            dtype (str): the data type of the series
        """
        self.name = name
        self.init = init
        self.dtype = dtype

    def make_series(self, len: int, back_end: BackendType) -> Series:
        """make a series

        Args:
            len (int): the number of elements
            back_end (BackendType): the backend type

        Returns:
            Series: initialized series
        """
        return allocate(self.name, len, self.init, self.dtype, back_end)


def allocate(
    name: str, len: int, init: Any, dtype: str, back_end: BackendType
) -> Series:
    """Allocate a series

    Args:
        name (str): series name
        len (int): number of elements
        init (Any): value to assign all elements
        dtype (str): data type
        back_end (BackendType): backend type

    Returns:
        Series: initialzed Series
    """
    return get_backend(back_end).allocate(name, len, init, dtype)


def range(
    name: str,
    start: int,
    stop: int,
    step: int,
    dtype: str,
    back_end: BackendType,
) -> Series:
    """Create a series using a range of values using the same methodolgy as
    `numpy.arange`

    Args:
        name (str): the name of the series
        start (int): start of interval
        stop (int): end of interval
        step (int): spacing between values
        dtype (str): data type
        back_end (BackendType): backend storage type

    Returns:
        Series: initialize series with range of values
    """
    return get_backend(back_end).range(name, start, stop, step, dtype)


def from_pandas(series: pd.Series, name: str = None) -> Series:
    """Initialize a series with a pandas Series.

    Args:
        series (pd.Series): the pandas Series
        name (str, optional): The name of the resulting series.  If
            unspecified and the specified pandas series has a defined name
            that is used in the result. Defaults to None.

    Returns:
        Series: the initialized series
    """
    return get_backend(BackendType.pandas).PandasSeriesBackend(
        name if name else series.name, series=series
    )


def from_numpy(name: str, data: np.ndarray) -> Series:
    """Initialize a series with a numpy array

    Args:
        name (str): the name of the resulting series
        data (np.ndarray): the series data

    Returns:
        Series: the initialized series
    """
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
