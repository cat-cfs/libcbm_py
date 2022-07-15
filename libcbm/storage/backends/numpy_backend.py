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


class _numepxr_local_dict_wrap:
    def __init__(self, col_idx: dict[str, int], arr: np.ndarray):
        self._arr = arr
        self._col_idx = col_idx

    def __getitem__(self, key: str) -> np.ndarray:
        return self._arr[:, self._col_idx[key]]


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
        self._data: np.ndarray = np.column_stack(list(data.values()))
        self._columns = list(data.keys())
        self._col_idx = {col: i for i, col in enumerate(self._columns)}
        self._n_rows: int = None
        self._n_cols: int = len(data)
        for k, v in data.items():
            if v.ndim != 1:
                raise ValueError(f"specified array '{k}' has ndim {v.ndim}")
            if self._n_rows is None:
                self._n_rows = v.shape[0]
            else:
                if self._n_rows != v.shape[0]:
                    raise ValueError("uneven array lengths")

    def __getitem__(self, col_name: str) -> Series:
        return NumpySeriesBackend(
            col_name, self._data[:, self._col_idx[col_name]]
        )

    def filter(self, arg: Series) -> DataFrame:
        _filter = arg.to_numpy()
        return NumpyDataFrameFrameBackend(
            {
                col: self._data[_filter, col_idx]
                for col, col_idx in self._col_idx.items()
            },
        )

    def take(self, indices: Series) -> DataFrame:
        row_idx = indices.to_numpy()
        return NumpyDataFrameFrameBackend(
            {
                col: self.data[row_idx, col_idx]
                for col, col_idx in self._col_idx.items()
            },
        )

    def at(self, index: int) -> dict:
        return {
            col: self._data[index, col_idx]
            for col, col_idx in self._col_idx.items()
        }

    def assign(
        self, col_name: str, value: Union[Series, Any], indices: Series = None
    ):
        assign_value = None
        if isinstance(value, Series):
            assign_value = value.to_numpy()
        else:
            assign_value = value
        if indices is not None:
            _idx = indices.to_numpy()
            self._data[_idx, self._col_idx[col_name]] = assign_value
        else:
            self._data[:, self._col_idx[col_name]] = assign_value

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def n_cols(self) -> int:
        return self._n_cols

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def backend_type(self) -> BackendType:
        return BackendType.numpy

    def copy(self) -> DataFrame:
        return NumpyDataFrameFrameBackend(
            {
                col: self._data[:, col_idx].copy()
                for col, col_idx in self._col_idx.items()
            }
        )

    def multiply(self, series: Series) -> DataFrame:
        rh = series.to_numpy()
        result = {
            col: self._data[:, col_idx] * rh
            for col, col_idx in self._col_idx.items()
        }
        return NumpyDataFrameFrameBackend(result)

    def add_column(self, series: Series, index: int) -> None:

        if series.name in self.columns:
            raise ValueError(
                f"{series.name} already present in this Dataframe"
            )
        insert_data = series.to_numpy()
        if insert_data.shape[0] != self.n_rows:
            raise ValueError(
                "specified series does not have the same length as the "
                "number of rows in this DataFrame"
            )
        self._data = np.insert(self._data, index, insert_data, axis=1)
        self._columns.insert(index, series.name)
        self._col_idx = {col: i for i, col in enumerate(self._columns)}
        self._n_cols: int = len(self._columns)

    def to_numpy(self, make_c_contiguous=True) -> np.ndarray:
        if make_c_contiguous and not self._data.flags["C_CONTIGUOUS"]:
            self._data = np.ascontiguousarray(self._data)
        return self._data

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self.columns, data=self._data)

    def zero(self):
        for v in self._data:
            v[:, :] = 0

    def map(self, arg: Union[dict, Callable]) -> DataFrame:
        out_data = {}
        for col_name, col_idx in self._col_idx:
            out_data[col_name] = (
                pd.Series(self._data[:, col_idx]).map(arg).to_numpy()
            )
        return NumpyDataFrameFrameBackend(out_data)

    def evaluate_filter(self, expression: str) -> Series:
        local_dict = _numepxr_local_dict_wrap(self._col_idx, self._data)
        return NumpySeriesBackend(
            None, numexpr.evaluate(expression, local_dict)
        )

    def sort_values(self, by: str, ascending: bool = True) -> "DataFrame":
        index_array = np.argsort(
            self._data[:, self._col_idx[by]], kind="mergesort"
        )
        return NumpyDataFrameFrameBackend(
            {
                col_name: self._data[index_array, col_idx]
                for col_name, col_idx in self._col_idx
            }
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

    @name.setter
    def name(self, value) -> str:
        self._name = value

    def copy(self):
        return NumpySeriesBackend(self._name, self._data.copy())

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

    def assign(
        self,
        indices: "Series",
        value: Union["Series", Any],
        allow_type_change=False,
    ):
        if allow_type_change:
            raise ValueError("numpy backend does not support type conversion")
        if isinstance(value, Series):
            self._data[indices.to_numpy()] = value.to_numpy()
        else:
            self._data[indices.to_numpy()] = value

    def assign_all(self, value: Union["Series", Any], allow_type_change=False):
        """
        set all values in this series to the specified value
        """
        if allow_type_change:
            raise ValueError("numpy backend does not support type conversion")
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

    def all(self) -> bool:
        """
        return True if all values in this series are non-zero
        """
        return self._data.all()

    def unique(self) -> "Series":
        return NumpySeriesBackend(self._name, np.unique(self._data))

    def to_numpy(self) -> np.ndarray:
        return self._data

    def to_list(self) -> list:
        return self._data.tolist()

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

    @property
    def data(self) -> np.ndarray:
        return self._data

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
        return NumpySeriesBackend(
            self._name, (self._data * self._get_operand(other))
        )

    def __rmul__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_operand(other) * self._data)
        )

    def __truediv__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._data / self._get_operand(other))
        )

    def __rtruediv__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_operand(other) / self._data)
        )

    def __add__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._data + self._get_operand(other))
        )

    def __radd__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_operand(other) + self._data)
        )

    def __sub__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._data - self._get_operand(other))
        )

    def __rsub__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_operand(other) - self._data)
        )

    def __ge__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._data >= self._get_operand(other))
        )

    def __gt__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._data > self._get_operand(other))
        )

    def __le__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._data <= self._get_operand(other))
        )

    def __lt__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._data < self._get_operand(other))
        )

    def __eq__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._data == self._get_operand(other))
        )

    def __ne__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._data != self._get_operand(other))
        )

    def __and__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._data & self._get_operand(other))
        )

    def __or__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._data | self._get_operand(other))
        )

    def __rand__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_operand(other) & self._data)
        )

    def __ror__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_operand(other) | self._data)
        )

    def __invert__(self) -> "Series":
        return NumpySeriesBackend(self._name, ~self._data)


def concat_data_frame(
    dfs: list[NumpyDataFrameFrameBackend],
) -> NumpyDataFrameFrameBackend:
    new_data = {}
    cols = []
    for df in dfs:
        if not cols:
            cols = list(df.columns)
        elif cols != list(df.columns):
            raise ValueError("cols do not match")

    new_data: dict[str, np.ndarray] = {}
    for col in cols:
        new_data[col] = np.concatenate(
            [df._data[:, df._col_idx[col]] for df in dfs]
        )

    return NumpyDataFrameFrameBackend(new_data)


def concat_series(series: list[NumpySeriesBackend]) -> NumpySeriesBackend:
    return NumpySeriesBackend(None, np.concatenate([s._data for s in series]))


def logical_and(
    s1: NumpySeriesBackend, s2: NumpySeriesBackend
) -> NumpySeriesBackend:
    return NumpySeriesBackend(None, np.logical_and(s1._data, s2._data))


def logical_not(series: NumpySeriesBackend) -> NumpySeriesBackend:
    return NumpySeriesBackend(None, np.logical_not(series._data))


def logical_or(
    s1: NumpySeriesBackend, s2: NumpySeriesBackend
) -> NumpySeriesBackend:
    return NumpySeriesBackend(None, np.logical_or(s1._data, s2._data))


def make_boolean_series(init: bool, size: int) -> NumpySeriesBackend:
    return NumpySeriesBackend(
        None, np.full(shape=size, fill_value=init, dtype="bool")
    )


def is_null(series: NumpySeriesBackend) -> NumpySeriesBackend:
    return NumpySeriesBackend(None, pd.isnull(series._data))


def indices_nonzero(series: NumpySeriesBackend) -> NumpySeriesBackend:
    return NumpySeriesBackend(None, np.nonzero(series._data)[0])


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


def from_series_dict(
    data: dict[str, NumpySeriesBackend],
) -> DataFrame:
    return NumpySeriesBackend(
        pd.DataFrame({k: v._data for k, v in data.items()})
    )


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
