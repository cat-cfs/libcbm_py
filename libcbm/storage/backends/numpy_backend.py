import ctypes
import numpy as np
import pandas as pd
import numexpr
from typing import Any
from typing import Union
from typing import Callable
from libcbm.storage.dataframe import DataFrame
from libcbm.storage.series import Series
from libcbm.storage.backends import BackendType


def get_numpy_pointer(
    data: np.ndarray, dtype=ctypes.c_double
) -> ctypes.pointer:
    """Helper method for wrapper parameters that can be specified either as
    null pointers or pointers to numpy memory.  Return a pointer to float64
    or int32 memory for use with ctypes wrapped functions, or None if None
    is specified.

    Args:
        data (numpy.ndarray, None): array to convert to pointer, if None is
            specified None is returned.
        type (object, optional): type supported by ctypes.POINTER. Defaults
            to ctypes.c_double.  Since libcbm only currently uses int32, or
            float 64, the only valid values are those that equal
            ctypes.c_double, or ctypes.c_int32

    Returns:
        None or ctypes.pointer: if the specified argument is None, None is
            returned, otherwise the argument is converted to a pointer to
            the underlying ndarray data.
    """
    if data is None:
        return None
    else:

        if not data.flags["C_CONTIGUOUS"]:
            raise ValueError("specified array is not C_CONTIGUOUS")
        if dtype == ctypes.c_double:
            if data.dtype != np.dtype("float64"):
                raise ValueError(
                    f"specified array is of type {data.dtype} "
                    f"and cannot be converted to {dtype}."
                )
        elif dtype == ctypes.c_int32:
            if data.dtype != np.dtype("int32"):
                raise ValueError(
                    f"specified array is of type {data.dtype} "
                    f"and cannot be converted to {dtype}."
                )
        else:
            raise ValueError(f"unsupported type {dtype}")
        p_result = data.ctypes.data_as(ctypes.POINTER(dtype))
        return p_result


class NumpyDataFrameFrameBackend(DataFrame):
    def __init__(self, data: dict[str, np.ndarray]) -> None:
        self._data = data
        self._n_rows: int = None
        self._n_cols: int = len(data)
        for k, v in self._data.items():
            if v.ndim != 1:
                raise ValueError(f"specified array '{k}' has ndim {v.ndim}")
            if self._n_rows is None:
                self._n_rows = v.shape[0]
            else:
                if self._n_rows != v.shape[0]:
                    raise ValueError("uneven array lengths")

    def __getitem__(self, col_name: str) -> Series:
        return NumpySeriesBackend(col_name, self._data[col_name])

    def filter(self, arg: Series) -> DataFrame:
        _filter = arg.to_numpy()
        return NumpyDataFrameFrameBackend(
            {k: v[_filter] for k, v in self._data.items()},
        )

    def take(self, indices: Series) -> DataFrame:
        _idx = indices.to_numpy()
        return NumpyDataFrameFrameBackend(
            {k: v[_idx] for k, v in self._data.items()},
        )

    def at(self, index: int) -> dict:
        return {k: v[index] for k, v in self._data.items()}

    def assign(self, col_name: str, value: Any, indices: Series = None):
        if indices is not None:
            _idx = indices.to_numpy()
            self._data[col_name][_idx] = value
        else:
            self._data[col_name][:] = value

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def n_cols(self) -> int:
        return self._n_cols

    @property
    def columns(self) -> list[str]:
        return list(self._data.keys())

    @property
    def backend_type(self) -> BackendType:
        return BackendType.numpy

    def copy(self) -> DataFrame:
        return NumpyDataFrameFrameBackend(
            {k: v.copy() for k, v in self._data.items()}
        )

    def multiply(self, series: Series) -> DataFrame:
        rh = series.to_numpy()
        result = {k: v * rh for k, v in self._data.items()}
        return NumpyDataFrameFrameBackend(result)

    def add_column(self, series: Series, index: int) -> None:
        if series.name in self._data:
            raise ValueError(
                f"{series.name} already present in this Dataframe"
            )
        data = series.to_numpy()
        if data.shape[0] != self.n_rows:
            raise ValueError(
                "specified series does not have the same length as the "
                "number of rows in this DataFrame"
            )
        if index == self.n_cols:
            self._data[series.name] = series.to_numpy()
        elif index >= 0:
            new_data = {}
            for i, (k, v) in enumerate(self._data.items()):
                if i == index:
                    new_data[series.name] = series.to_numpy()
                else:
                    new_data[k] = v
            self._data = new_data
        else:
            raise ValueError("index out of range")
        self._df.insert(index, series.name, series.to_numpy())

    def to_c_contiguous_numpy_array(self) -> np.ndarray:
        return np.ascontiguousarray(np.column_stack(list(self._data.values())))

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self._data)

    def zero(self):
        for v in self._data:
            v[:] = 0

    def map(self, arg: Union[dict, Callable]) -> DataFrame:
        out_data = {}
        for k, v in self._data.items():
            out_data[k] = NumpySeriesBackend(k, v).map(arg).to_numpy()
        return NumpyDataFrameFrameBackend(out_data)

    def evaluate_filter(self, expression: str) -> Series:
        return NumpySeriesBackend(
            None, numexpr.evaluate(expression, self._data)
        )


class NumpySeriesBackend(Series):
    """
    Series is a wrapper for one of several underlying storage types which
    presents a limited interface for internal usage by libcbm.
    """

    def __init__(self, name: str, data: np.ndarray):
        self._name = name
        self._data = data

    @property
    def name(self) -> str:
        return self._name

    def filter(self, arg: "Series") -> "Series":
        """
        Return a new series of the elements
        corresponding to the true values in the specified arg
        """
        return NumpySeriesBackend(self._name, self._data[arg.to_numpy()])

    def take(self, indices: "Series") -> "Series":
        """return the elements of this series at the specified indices
        (returns a copy)"""
        return NumpySeriesBackend(
            self._name,
            self._data[indices.to_numpy()],
        )

    def as_type(self, type_name: str) -> "Series":
        return NumpySeriesBackend(self._name, self._data.astype(type_name))

    def assign(self, indices: "Series", value: Union["Series", Any]):
        if isinstance(value, Series):
            self._data[indices.to_numpy()] = value.to_numpy()
        else:
            self._data[indices.to_numpy()] = value

    def assign_all(self, value: Union["Series", Any]):
        """
        set all values in this series to the specified value
        """
        if isinstance(value, Series):
            self._data[:] = value.to_numpy()
        else:
            self._data[:] = value

    def map(self, arg: Union[dict, Callable[[int, Any], Any]]) -> "Series":
        return NumpySeriesBackend(
            self._name, pd.Series(self._data).map(arg).to_numpy()
        )

    def at(self, idx: int) -> Any:
        """Gets the value at the specified sequential index"""
        return self._data[idx]

    def any(self) -> bool:
        """
        return True if at least one value in this series is
        non-zero
        """
        return self._data.any()

    def unique(self) -> "Series":
        return NumpySeriesBackend(self._name, np.unique(self._data))

    def to_numpy(self) -> np.ndarray:
        return self._data

    def to_numpy_ptr(self) -> ctypes.pointer:
        if str(self._data.dtype) == "int32":
            ptr_type = ctypes.c_int32
        elif str(self._data.dtype) == "float64":
            ptr_type = ctypes.c_double
        else:
            raise ValueError(
                f"series type not supported {str(self._data.dtype)}"
            )
        return get_numpy_pointer(self._data, ptr_type)

    def less(self, other: "Series") -> "Series":
        """
        returns a boolean series with:
            True - where this series is less than the other series
            False - where this series is greater than or equal to the other
                series
        """
        return self._data < other.to_numpy()

    def sum(self) -> Union[int, float]:
        return self._data.sum()

    def cumsum(self) -> "NumpySeriesBackend":
        return NumpySeriesBackend(self.name, self._data.cumsum())

    def max(self) -> Union[int, float]:
        return self._data.max()

    def min(self) -> Union[int, float]:
        return self._data.min()

    @property
    def length(self) -> int:
        return self._data.size

    @property
    def backend_type(self) -> BackendType:
        return BackendType.numpy

    def __mul__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(self._name, (self._data * other))

    def __rmul__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(self._name, (other * self._data))

    def __add__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(self._name, (other + self._data))

    def __radd__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(self._name, (other + self._data))

    def __ge__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(self._name, (other >= self._data))

    def __gt__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(self._name, (other > self._data))

    def __le__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(self._name, (other <= self._data))

    def __lt__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(self._name, (other < self._data))

    def __eq__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(self._name, (other == self._data))


def concat_data_frame(
    dfs: list[NumpyDataFrameFrameBackend],
) -> NumpyDataFrameFrameBackend:
    new_data = {}
    cols = []
    for df in dfs:
        if not cols:
            cols = list(df._data.keys())
        elif cols != list(df._data.keys()):
            raise ValueError("cols do not match")

    for name in cols:
        new_data[name] = np.concatenate([df._data[name] for df in dfs])
    return NumpyDataFrameFrameBackend(new_data)


def concat_series(series: list[NumpySeriesBackend]) -> NumpySeriesBackend:
    return NumpySeriesBackend(None, np.concatenate([s._data for s in series]))


def logical_and(
    s1: NumpySeriesBackend, s2: NumpySeriesBackend
) -> NumpySeriesBackend:
    return NumpySeriesBackend(None, np.logical_and([s1._data, s2._data]))


def logical_not(series: NumpySeriesBackend) -> NumpySeriesBackend:
    return NumpySeriesBackend(None, np.logical_not(series._data))


def logical_or(
    s1: NumpySeriesBackend, s2: NumpySeriesBackend
) -> NumpySeriesBackend:
    return NumpySeriesBackend(None, np.logical_or([s1._data, s2._data]))


def make_boolean_series(init: bool, size: int) -> NumpySeriesBackend:
    return NumpySeriesBackend(
        None, np.full(shape=size, fill_value=init, dtype="bool")
    )


def is_null(series: NumpySeriesBackend) -> NumpySeriesBackend:
    return NumpySeriesBackend(None, pd.isnull(series._data))


def indices_nonzero(series: NumpySeriesBackend) -> NumpySeriesBackend:
    return NumpySeriesBackend(None, pd.isnull(series._data))


def numeric_dataframe(
    cols: list[str],
    nrows: int,
    init: float = 0.0,
) -> NumpyDataFrameFrameBackend:
    return NumpyDataFrameFrameBackend(
        {col: np.full(nrows, init, "float64") for col in cols}
    )


def from_series_list(
    series_list: list[NumpySeriesBackend],
) -> NumpyDataFrameFrameBackend:
    return NumpyDataFrameFrameBackend({s.name: s._data for s in series_list})


def allocate(name: str, len: int, init: Any, dtype: str) -> NumpySeriesBackend:
    return NumpySeriesBackend(name, np.full(len, init, dtype))


def range(
    name: str,
    start: int,
    stop: int,
    step: int,
    dtype: str,
) -> Series:
    return NumpySeriesBackend(
        name, np.arange(start=start, stop=stop, step=step, dtype=dtype)
    )
