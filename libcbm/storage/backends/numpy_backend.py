from __future__ import annotations
from enum import Enum
import ctypes
import numpy as np
import pandas as pd
import numexpr
from typing import Any
from typing import Union
from libcbm.storage.dataframe import DataFrame
from libcbm.storage.series import Series
from libcbm.storage.backends import BackendType


class StorageFormat(Enum):
    uniform_matrix = 0
    mixed_columns = 1


class _numepxr_local_dict_wrap:
    def __init__(self, col_idx: dict[str, int], arr: np.ndarray):
        self._arr = arr
        self._col_idx = col_idx

    def __getitem__(self, key: str) -> np.ndarray:
        return self._arr[:, self._col_idx[key]]


def _map_1D_nb(a: np.ndarray, out: np.ndarray, d: dict) -> np.ndarray:
    for i in np.arange(a.shape[0]):
        out[i] = d[a[i]]


def _map_2D_nb(a: np.ndarray, out: np.ndarray, d: dict) -> np.ndarray:
    for i in np.arange(a.shape[0]):
        for j in np.arange(a.shape[1]):
            out[i, j] = d[a[i, j]]


def _get_map_value_type(d: dict) -> str:
    out_value_type = type(next(iter(d.values())))
    if out_value_type == str:
        out_value_type = "object"
    return out_value_type


def _map(a: np.ndarray, d: dict) -> np.ndarray:
    if a.size == 0:
        if len(d) > 0:
            return a.astype(_get_map_value_type(d))
        else:
            return a.copy()
    elif len(d) == 0:
        raise ValueError("empty dictionary provided")

    out = np.empty_like(a, dtype=_get_map_value_type(d))

    if a.ndim == 1:
        _map_1D_nb(a, out, d)
    elif a.ndim == 2:
        _map_2D_nb(a, out, d)
    else:
        raise ValueError("ndim=1 or ndim=2 supported")
    return out


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
    def __init__(
        self,
        data: Union[np.ndarray, dict[str, np.ndarray]],
        cols: list[str] = None,
    ) -> None:
        if isinstance(data, dict):
            self._from_dict(data)
        else:
            self._from_matrix(cols, data)

    def _initialize(self, n_rows: int, columns: list[str]):
        self._columns = columns
        self._col_idx = {col: i for i, col in enumerate(self._columns)}
        self._n_rows: int = n_rows
        self._n_cols: int = len(columns)

    def _from_matrix(self, cols: list[str], data: np.ndarray) -> None:
        self._data_matrix: np.ndarray = data
        self._data_cols: dict[str, np.ndarray] = None
        self._storage_format = StorageFormat.uniform_matrix
        self._initialize(data.shape[0], cols)

    def _from_dict(self, data: dict[str, np.ndarray]) -> None:
        n_rows = None
        for k, v in data.items():
            if v.ndim != 1:
                raise ValueError(f"specified array '{k}' has ndim {v.ndim}")
            if n_rows is None:
                n_rows = v.shape[0]
            else:
                if n_rows != v.shape[0]:
                    raise ValueError("uneven array lengths")

        _has_uniform_types = len(set(arr.dtype for arr in data.values())) == 1
        if _has_uniform_types:
            self._storage_format = StorageFormat.uniform_matrix
            self._data_matrix: np.ndarray = np.column_stack(
                list(data.values())
            )
            self._data_cols: dict[str, np.ndarray] = None
        else:
            self._storage_format = StorageFormat.mixed_columns
            self._data_matrix: np.ndarray = None
            self._data_cols: dict[str, np.ndarray] = data

        self._initialize(n_rows, list(data.keys()))

    def __getitem__(self, col_name: str) -> Series:
        return NumpySeriesBackend(col_name, parent_df=self)

    def filter(self, arg: Series) -> DataFrame:
        _filter = arg.to_numpy()
        if self._storage_format == StorageFormat.uniform_matrix:
            return NumpyDataFrameFrameBackend(
                self._data_matrix[_filter, :], self.columns
            )
        else:
            return NumpyDataFrameFrameBackend(
                {col: self._data_cols[col][_filter] for col in self._columns}
            )

    def take(self, indices: Series) -> DataFrame:
        row_idx = indices.to_numpy()
        if self._storage_format == StorageFormat.uniform_matrix:
            return NumpyDataFrameFrameBackend(
                self._data_matrix[row_idx, :], self.columns
            )
        else:
            return NumpyDataFrameFrameBackend(
                {col: self._data_cols[col][row_idx] for col in self._columns}
            )

    def at(self, index: int) -> dict:
        if self._storage_format == StorageFormat.uniform_matrix:
            return {
                col: self._data_matrix[index, col_idx]
                for col, col_idx in self._col_idx.items()
            }
        else:
            return {col: self._data_cols[col][index] for col in self.columns}

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
        if self._storage_format == StorageFormat.uniform_matrix:
            return NumpyDataFrameFrameBackend(
                self._data_matrix.copy(), self.columns
            )
        else:
            return NumpyDataFrameFrameBackend(
                {col: self._data_cols[col].copy() for col in self.columns}
            )

    def multiply(self, series: Series) -> DataFrame:
        rh = series.to_numpy()
        if self._storage_format == StorageFormat.uniform_matrix:
            return NumpyDataFrameFrameBackend(
                np.column_stack(
                    [
                        self._data_matrix[:, idx] * rh
                        for _, idx in self._col_idx.items()
                    ]
                ),
                cols=self.columns,
            )
        else:
            return NumpyDataFrameFrameBackend(
                {
                    col: self._data[:, col_idx] * rh
                    for col, col_idx in self._col_idx.items()
                }
            )

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

        if self._storage_format == StorageFormat.uniform_matrix:
            if insert_data.dtype == self._data_matrix.dtype:
                self._data_matrix = np.insert(
                    self._data_matrix, index, insert_data, axis=1
                )
            else:
                self._storage_format = StorageFormat.mixed_columns
                self._data_cols = {
                    col: np.ascontiguousarray(self._data_matrix[:, idx])
                    for col, idx in self._col_idx.items()
                }
                self._data_cols[series.name] = series.to_numpy()
                self._data_matrix = None

        else:
            self._data_cols[series.name] = insert_data

        self._columns.insert(index, series.name)
        self._col_idx = {col: i for i, col in enumerate(self._columns)}
        self._n_cols: int = len(self._columns)

    def to_numpy(self, make_c_contiguous=True) -> np.ndarray:
        if self._storage_format != StorageFormat.uniform_matrix:
            raise ValueError("to_numpy not supported for non-uniform matrix")
        if make_c_contiguous and not self._data_matrix.flags["C_CONTIGUOUS"]:
            self._data_matrix = np.ascontiguousarray(self._data_matrix)
        return self._data_matrix

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=self.columns,
            data=(
                self._data_matrix
                if self._storage_format == StorageFormat.uniform_matrix
                else self._data_cols
            ),
        )

    def zero(self):
        if self._storage_format == StorageFormat.uniform_matrix:
            self._data_matrix[:, :] = 0
        else:
            for col in self.columns:
                self._data_cols[col][:] = 0

    def map(self, arg: dict) -> DataFrame:
        if self._storage_format == StorageFormat.uniform_matrix:
            return NumpyDataFrameFrameBackend(
                _map(self._data_matrix, arg), self.columns
            )
        else:
            return NumpyDataFrameFrameBackend(
                {col: _map(self._data_cols[col], arg) for col in self.columns}
            )

    def evaluate_filter(self, expression: str) -> Series:
        if self._storage_format == StorageFormat.uniform_matrix:
            local_dict = _numepxr_local_dict_wrap(
                self._col_idx, self._data_matrix
            )
            return NumpySeriesBackend(
                None, numexpr.evaluate(expression, local_dict)
            )
        else:
            return NumpySeriesBackend(
                None, numexpr.evaluate(expression, self._data_cols)
            )

    def sort_values(self, by: str, ascending: bool = True) -> "DataFrame":
        sort_col = (
            self._data_matrix[:, self._col_idx[by]]
            if self._storage_format == StorageFormat.uniform_matrix
            else self._data_cols[by]
        )
        index_array = np.argsort(sort_col, kind="mergesort")

        if not ascending:
            index_array_slice = slice(None, None, -1)
            index_array = index_array[index_array_slice]

        if self._storage_format == StorageFormat.uniform_matrix:
            return NumpyDataFrameFrameBackend(
                self._data_matrix[index_array, :], cols=self.columns
            )
        else:
            return NumpyDataFrameFrameBackend(
                {
                    col_name: self._data_cols[col_name][index_array]
                    for col_name in self.columns
                }
            )


class NumpySeriesBackend(Series):
    """
    Series is a wrapper for one of several underlying storage types which
    presents a limited interface for internal usage by libcbm.
    """

    def __init__(
        self,
        name: str,
        data: np.ndarray = None,
        parent_df: NumpyDataFrameFrameBackend = None,
    ):
        if not ((data is None) ^ (parent_df is None)):
            raise ValueError("one of data, or parent_df must be specified")

        self._name = name
        self._data = data
        self._parent_df = parent_df

    def _get_dtype(self) -> str:
        if self._data is not None:
            return str(self._data.dtype)
        else:
            if self._parent_df._storage_format == StorageFormat.uniform_matrix:
                return str(self._parent_df._data_matrix.dtype)
            else:
                return str(self._parent_df._data_cols[self.name].dtype)

    def _get_data(self, reference_required=False) -> np.ndarray:
        if self._data is not None:
            return self._data
        else:
            if self._parent_df._storage_format == StorageFormat.uniform_matrix:
                if reference_required:
                    self._parent_df._data_cols = {
                        col: np.ascontiguousarray(
                            self._parent_df._data_matrix[
                                :,
                                self._parent_df._col_idx[col],
                            ]
                        )
                        for col in self._parent_df.columns
                    }
                    self._parent_df._data_matrix = None
                    self._parent_df._storage_format = (
                        StorageFormat.mixed_columns
                    )
                    return self._parent_df._data_cols[self.name]
                else:
                    return self._parent_df._data_matrix[
                        :, self._parent_df._col_idx[self.name]
                    ]
            else:
                return self._parent_df._data_cols[self.name]

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value) -> str:
        self._name = value

    def copy(self):
        return NumpySeriesBackend(self._name, self._get_data().copy())

    def filter(self, arg: "Series") -> "Series":
        """
        Return a new series of the elements
        corresponding to the true values in the specified arg
        """
        return NumpySeriesBackend(self._name, self._get_data()[arg.to_numpy()])

    def take(self, indices: "Series") -> "Series":
        """return the elements of this series at the specified indices
        (returns a copy)"""
        return NumpySeriesBackend(
            self._name,
            self._get_data()[indices.to_numpy()],
        )

    def is_null(self) -> "Series":
        return NumpySeriesBackend(self._name, pd.isnull(self._get_data()))

    def as_type(self, type_name: str) -> "Series":
        return NumpySeriesBackend(
            self._name, self._get_data().astype(type_name)
        )

    def assign(
        self,
        value: Union["Series", Any],
        indices: "Series" = None,
        allow_type_change=False,
    ):
        assignment_value = None
        if isinstance(value, Series):
            assignment_value = value.to_numpy()
        else:
            assignment_value = np.array(value)

        dtype_original = self._get_data().dtype

        if self._data is not None:
            if indices is not None:
                self._data[indices.to_numpy()] = assignment_value
            else:
                self._data = np.full(
                    self._data.shape,
                    assignment_value,
                    dtype=(
                        self._data.dtype if not allow_type_change else None
                    ),
                )

        elif self._parent_df is not None:
            if indices is not None:
                _idx = indices.to_numpy()
            else:
                _idx = slice(None)

            if self._parent_df._storage_format == StorageFormat.uniform_matrix:
                if (
                    assignment_value.dtype
                    != self._parent_df._data_matrix.dtype
                ):
                    if not allow_type_change:
                        raise ValueError("type change not allowed")
                    self._parent_df._data_cols = {
                        col: np.ascontiguousarray(
                            self._parent_df._data_matrix[
                                _idx,
                                self._parent_df._col_idx[col],
                            ]
                        )
                        for col in self._parent_df.columns
                    }
                    self._parent_df._data_matrix = None
                    self._parent_df._storage_format = (
                        StorageFormat.mixed_columns
                    )
                else:
                    self._parent_df._data_matrix[
                        _idx,
                        self._parent_df._col_idx[self.name],
                    ] = assignment_value
            else:
                if indices is not None:
                    self._parent_df._data_cols[self.name][
                        _idx
                    ] = assignment_value
                else:
                    self._parent_df._data_cols[self.name] = np.full(
                        self._parent_df.n_rows, assignment_value
                    )

        else:
            raise ValueError("internal series not defined")

        if not allow_type_change and dtype_original != self._get_data().dtype:
            raise ValueError("type change not allowed")

    def map(self, arg: dict) -> "Series":
        return NumpySeriesBackend(self._name, _map(self._get_data(), arg))

    def at(self, idx: int) -> Any:
        """Gets the value at the specified sequential index"""
        return self._get_data()[idx]

    def any(self) -> bool:
        """
        return True if at least one value in this series is
        non-zero
        """
        return self._get_data().any()

    def all(self) -> bool:
        """
        return True if all values in this series are non-zero
        """
        return self._get_data().all()

    def indices_nonzero(self) -> "Series":
        """Get the indices of values that are non-zero in this series"""
        return NumpySeriesBackend(self._name, np.nonzero(self._get_data())[0])

    def unique(self) -> "Series":
        return NumpySeriesBackend(self._name, np.unique(self._get_data()))

    def to_numpy(self) -> np.ndarray:
        return self._get_data(reference_required=True)

    def to_list(self) -> list:
        return self._get_data().tolist()

    def to_numpy_ptr(self) -> ctypes.pointer:
        dtype = self._get_dtype()
        if dtype == "int32":
            ptr_type = ctypes.c_int32
        elif dtype == "float64":
            ptr_type = ctypes.c_double
        else:
            raise ValueError(f"series type not supported {dtype}")
        return get_numpy_pointer(
            self._get_data(reference_required=True), ptr_type
        )

    @property
    def data(self) -> np.ndarray:
        return self._get_data()

    def sum(self) -> Union[int, float]:
        return self._get_data().sum()

    def cumsum(self) -> "NumpySeriesBackend":
        return NumpySeriesBackend(self.name, self._get_data().cumsum())

    def max(self) -> Union[int, float]:
        return self._get_data().max()

    def min(self) -> Union[int, float]:
        return self._get_data().min()

    @property
    def length(self) -> int:
        return self._get_data().size

    @property
    def backend_type(self) -> BackendType:
        return BackendType.numpy

    def __mul__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_data() * self._get_operand(other))
        )

    def __rmul__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_operand(other) * self._data)
        )

    def __truediv__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_data() / self._get_operand(other))
        )

    def __rtruediv__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_operand(other) / self._get_data())
        )

    def __add__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_data() + self._get_operand(other))
        )

    def __radd__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_operand(other) + self._get_data())
        )

    def __sub__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_data() - self._get_operand(other))
        )

    def __rsub__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_operand(other) - self._get_data())
        )

    def __ge__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_data() >= self._get_operand(other))
        )

    def __gt__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_data() > self._get_operand(other))
        )

    def __le__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_data() <= self._get_operand(other))
        )

    def __lt__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_data() < self._get_operand(other))
        )

    def __eq__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_data() == self._get_operand(other))
        )

    def __ne__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_data() != self._get_operand(other))
        )

    def __and__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_data() & self._get_operand(other))
        )

    def __or__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_data() | self._get_operand(other))
        )

    def __rand__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_operand(other) & self._get_data())
        )

    def __ror__(self, other: Union[int, float, "Series"]) -> "Series":
        return NumpySeriesBackend(
            self._name, (self._get_operand(other) | self._get_data())
        )

    def __invert__(self) -> "Series":
        return NumpySeriesBackend(self._name, ~self._get_data())


def concat_data_frame(
    dfs: list[NumpyDataFrameFrameBackend],
) -> NumpyDataFrameFrameBackend:
    new_data = {}
    cols = []
    for df in dfs:
        if not cols:
            cols = list(df.columns)
        elif cols != df.columns:
            raise ValueError("cols do not match")

    storage_format_set = set([x._storage_format for x in dfs])

    if (
        len(storage_format_set) == 1
        and StorageFormat.uniform_matrix in storage_format_set
    ):
        # case 1, all incoming df's are uniform_matrix
        matrix_data = np.concatenate(
            [df._data_matrix for df in dfs], axis=0, casting="no"
        )
        return NumpyDataFrameFrameBackend(matrix_data, cols)
    else:
        # case 2, all incoming df's are mixed_columns, or a mix of
        # mixed_columns, uniform_matrix
        new_data: dict[str, np.ndarray] = {}

        for col in cols:
            concat_list = []
            for df in dfs:
                ser = df[col]
                concat_list.append(ser.to_numpy())
            new_data[col] = np.concatenate(concat_list, casting="no")

        return NumpyDataFrameFrameBackend(new_data)


def concat_series(series: list[NumpySeriesBackend]) -> NumpySeriesBackend:
    return NumpySeriesBackend(
        None, np.concatenate([s._get_data() for s in series])
    )


def logical_and(
    s1: NumpySeriesBackend, s2: NumpySeriesBackend
) -> NumpySeriesBackend:
    return NumpySeriesBackend(
        None, np.logical_and(s1._get_data(), s2._get_data())
    )


def logical_not(series: NumpySeriesBackend) -> NumpySeriesBackend:
    return NumpySeriesBackend(None, np.logical_not(series._get_data()))


def logical_or(
    s1: NumpySeriesBackend, s2: NumpySeriesBackend
) -> NumpySeriesBackend:
    return NumpySeriesBackend(
        None, np.logical_or(s1._get_data(), s2._get_data())
    )


def make_boolean_series(init: bool, size: int) -> NumpySeriesBackend:
    return NumpySeriesBackend(
        None, np.full(shape=size, fill_value=init, dtype="bool")
    )


def is_null(series: NumpySeriesBackend) -> NumpySeriesBackend:
    return NumpySeriesBackend(None, pd.isnull(series._get_data()))


def indices_nonzero(series: NumpySeriesBackend) -> NumpySeriesBackend:
    return NumpySeriesBackend(None, np.nonzero(series._get_data())[0])


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
    return NumpyDataFrameFrameBackend(
        {s.name: s._get_data() for s in series_list}
    )


def from_series_dict(
    data: dict[str, NumpySeriesBackend],
) -> DataFrame:
    return NumpySeriesBackend(
        pd.DataFrame({k: v._get_data() for k, v in data.items()})
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
