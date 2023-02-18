from __future__ import annotations
from typing import Any
from typing import Union
import pandas as pd
import numpy as np
import ctypes
import numexpr
from libcbm.storage.dataframe import DataFrame
from libcbm.storage.series import Series
from libcbm.storage.backends import BackendType
from libcbm.storage.backends import numpy_backend


class PandasDataFrameBackend(DataFrame):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def __getitem__(self, col_name: str) -> Series:
        return PandasSeriesBackend(col_name, parent_df=self._df)

    def filter(self, arg: Series) -> DataFrame:
        return PandasDataFrameBackend(
            self._df[arg.to_numpy()].reset_index(drop=True)
        )

    def take(self, indices: Series) -> DataFrame:
        return PandasDataFrameBackend(
            self._df.iloc[indices.to_numpy()].reset_index(drop=True)
        )

    def at(self, index: int) -> dict:
        return self._df.iloc[index].to_dict()

    @property
    def n_rows(self) -> int:
        return len(self._df.index)

    @property
    def n_cols(self) -> int:
        return len(self._df.columns)

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    @property
    def backend_type(self) -> BackendType:
        return BackendType.pandas

    def copy(self) -> DataFrame:
        return PandasDataFrameBackend(self._df.copy())

    def multiply(self, series: Series) -> DataFrame:
        result = self._df.multiply(series.to_numpy(), axis=0)
        return PandasDataFrameBackend(result)

    def add_column(self, series: Series, index: int) -> None:
        self._df.insert(index, series.name, series.to_numpy())

    def to_numpy(self, make_c_contiguous=True) -> np.ndarray:
        if make_c_contiguous and not self._df.values.flags["C_CONTIGUOUS"]:
            self._df = pd.DataFrame(
                index=self._df.index,
                columns=list(self._df.columns),
                data=np.ascontiguousarray(self._df.values),
            )
        return self._df.values

    def to_pandas(self) -> pd.DataFrame:
        return self._df

    def zero(self):
        self._df.iloc[:] = 0

    def map(self, arg: dict) -> DataFrame:
        cols = list(self._df.columns)
        output = pd.DataFrame(
            index=self._df.index,
            columns=cols,
            data={col: self[col].map(arg)._get_series() for col in cols},
        )
        return PandasDataFrameBackend(output)

    def evaluate_filter(self, expression: str) -> Series:
        return PandasSeriesBackend(
            None, pd.Series(numexpr.evaluate(expression, self._df))
        )

    def sort_values(self, by: str, ascending: bool = True) -> "DataFrame":
        return PandasDataFrameBackend(
            self._df.sort_values(by=by, ascending=ascending, kind="mergesort")
        )


class PandasSeriesBackend(Series):
    """
    Series is a wrapper for one of several underlying storage types which
    presents a limited interface for internal usage by libcbm.
    """

    def __init__(
        self,
        name: str,
        series: pd.Series = None,
        parent_df: pd.DataFrame = None,
    ):
        self._name = name
        if not ((series is None) ^ (parent_df is None)):
            raise ValueError("one of series, or parent_df must be specified")
        self._series = series
        self._parent_df = parent_df

    def _get_series(self) -> pd.Series:
        if self._series is not None:
            return self._series
        else:
            return self._parent_df[self._name]

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value) -> str:
        self._name = value

    def copy(self):
        return PandasSeriesBackend(self._name, self._get_series().copy())

    def filter(self, arg: "Series") -> "Series":
        """
        Return a new series of the elements
        corresponding to the true values in the specified arg
        """
        return PandasSeriesBackend(
            self._name,
            self._get_series().loc[arg.to_numpy()].reset_index(drop=True),
        )

    def take(self, indices: "Series") -> "Series":
        """return the elements of this series at the specified indices
        (returns a copy)"""
        return PandasSeriesBackend(
            self._name,
            self._get_series().iloc[indices.to_numpy()].reset_index(drop=True),
        )

    def is_null(self) -> "Series":
        return PandasSeriesBackend(self._name, pd.isnull(self._get_series()))

    def as_type(self, type_name: str) -> "Series":
        return PandasSeriesBackend(
            self._name, self._get_series().astype(type_name)
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
            assignment_value = value

        dtype_original = self._get_series().dtype
        if indices is not None:
            _idx = indices.to_numpy()
        else:
            _idx = slice(None)
        if self._series is not None:

            self._series.iloc[_idx] = assignment_value

            if (
                not allow_type_change
                and dtype_original != self._get_series().dtype
            ):
                self._series = self._series.astype(dtype_original)
        elif self._parent_df is not None:

            self._parent_df.iloc[
                _idx,
                self._parent_df.columns.get_loc(self.name),
            ] = assignment_value

            if (
                not allow_type_change
                and dtype_original != self._get_series().dtype
            ):
                self._parent_df[self.name] = self._parent_df[self.name].astype(
                    dtype_original
                )
        else:
            raise ValueError("internal series not defined")

    def map(self, arg: dict) -> "Series":
        this_series = self._get_series()
        if len(this_series) == 0:
            return self.copy()
        elif len(arg) == 0:
            raise ValueError("specified map is empty")
        # This check may be built into pandas eventually as well
        # https://github.com/pandas-dev/pandas/issues/14210
        if len(
            set().union(arg.keys(), this_series.drop_duplicates())
        ) > len(arg.keys()):
            raise KeyError(
                "values in array not found as keys in specified dictionary"
            )
        return PandasSeriesBackend(self._name, this_series.map(arg))

    def at(self, idx: int) -> Any:
        """Gets the value at the specified sequential index"""
        return self._get_series().iloc[idx]

    def any(self) -> bool:
        """
        return True if at least one value in this series is
        non-zero
        """
        return self._get_series().any()

    def all(self) -> bool:
        """
        return True if all values in this series are non-zero
        """
        return self._get_series().all()

    def unique(self) -> "Series":
        return PandasSeriesBackend(
            self._name,
            pd.Series(name=self._name, data=self._get_series().unique()),
        )

    def to_numpy(self) -> np.ndarray:
        return self._get_series().values

    def to_list(self) -> list:
        return self._get_series().to_list()

    def to_numpy_ptr(self) -> ctypes.pointer:
        _dtype = str(self._get_series().dtype)
        if _dtype == "int32":
            ptr_type = ctypes.c_int32
        elif _dtype == "float64":
            ptr_type = ctypes.c_double
        else:
            raise ValueError(f"series type not supported {_dtype}")
        return numpy_backend.get_numpy_pointer(
            self._get_series().values, ptr_type
        )

    @property
    def data(self) -> pd.Series:
        return self._get_series()

    def sum(self) -> Union[int, float]:
        return self._get_series().sum()

    def cumsum(self) -> "PandasSeriesBackend":
        return PandasSeriesBackend(self.name, self._get_series().cumsum())

    def max(self) -> Union[int, float]:
        return self._get_series().max()

    def min(self) -> Union[int, float]:
        return self._get_series().min()

    @property
    def length(self) -> int:
        return self._get_series().size

    @property
    def backend_type(self) -> BackendType:
        return BackendType.pandas

    def __mul__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_series() * self._get_operand(other))
        )

    def __rmul__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_operand(other) * self._get_series())
        )

    def __truediv__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_series() / self._get_operand(other))
        )

    def __rtruediv__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_operand(other) / self._get_series())
        )

    def __add__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_series() + self._get_operand(other))
        )

    def __radd__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_operand(other) + self._get_series())
        )

    def __sub__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_series() - self._get_operand(other))
        )

    def __rsub__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_operand(other) - self._get_series())
        )

    def __ge__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_series() >= self._get_operand(other))
        )

    def __gt__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_series() > self._get_operand(other))
        )

    def __le__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_series() <= self._get_operand(other))
        )

    def __lt__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_series() < self._get_operand(other))
        )

    def __eq__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_series() == self._get_operand(other))
        )

    def __ne__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_series() != self._get_operand(other))
        )

    def __and__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_series() & self._get_operand(other))
        )

    def __or__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_series() | self._get_operand(other))
        )

    def __rand__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_operand(other) & self._get_series())
        )

    def __ror__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._get_operand(other) | self._get_series())
        )

    def __invert__(self) -> "Series":
        return PandasSeriesBackend(self._name, ~self._get_series())


def concat_data_frame(
    dfs: list[PandasDataFrameBackend],
) -> PandasDataFrameBackend:
    return PandasDataFrameBackend(
        pd.concat([d._df for d in dfs], ignore_index=True)
    )


def concat_series(series: list[PandasSeriesBackend]) -> PandasSeriesBackend:
    return PandasSeriesBackend(
        None, pd.concat([s._get_series() for s in series])
    )


def logical_and(
    s1: PandasSeriesBackend, s2: PandasSeriesBackend
) -> PandasSeriesBackend:
    return PandasSeriesBackend(None, s1._get_series() & s2._get_series())


def logical_not(series: PandasSeriesBackend) -> PandasSeriesBackend:
    return PandasSeriesBackend(None, ~(series._get_series()))


def logical_or(
    s1: PandasSeriesBackend, s2: PandasSeriesBackend
) -> PandasSeriesBackend:
    return PandasSeriesBackend(None, s1._get_series() | s2._get_series())


def make_boolean_series(init: bool, size: int) -> PandasSeriesBackend:
    return PandasSeriesBackend(
        None, pd.Series(np.full(shape=size, fill_value=init, dtype="bool"))
    )


def is_null(series: PandasSeriesBackend) -> PandasSeriesBackend:
    return PandasSeriesBackend(
        None, pd.Series(pd.isnull(series._get_series()))
    )


def indices_nonzero(series: PandasSeriesBackend) -> PandasSeriesBackend:
    return PandasSeriesBackend(
        None, pd.Series(series._get_series().to_numpy().nonzero()[0])
    )


def numeric_dataframe(
    cols: list[str],
    nrows: int,
    init: float = 0.0,
) -> PandasDataFrameBackend:
    return PandasDataFrameBackend(
        pd.DataFrame(
            columns=cols, data=np.full((nrows, len(cols)), init, "float64")
        )
    )


def from_series_list(
    series_list: list[PandasSeriesBackend],
) -> PandasDataFrameBackend:
    return PandasDataFrameBackend(
        pd.DataFrame({s.name: s._get_series() for s in series_list})
    )


def from_series_dict(
    data: dict[str, PandasSeriesBackend],
) -> DataFrame:
    return PandasDataFrameBackend(
        pd.DataFrame({k: v._get_series() for k, v in data.items()})
    )


def allocate(
    name: str, len: int, init: Any, dtype: str
) -> PandasSeriesBackend:
    return PandasSeriesBackend(name, pd.Series(np.full(len, init, dtype)))


def range(
    name: str,
    start: int,
    stop: int,
    step: int,
    dtype: str,
) -> Series:
    return PandasSeriesBackend(
        name,
        pd.Series(np.arange(start=start, stop=stop, step=step, dtype=dtype)),
    )
