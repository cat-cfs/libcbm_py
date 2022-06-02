from typing import Any
from typing import Union
from typing import Callable
import pandas as pd
import numpy as np
import ctypes

from libcbm.storage.dataframe import DataFrame
from libcbm.storage.series import Series
from libcbm.storage.backends import BackendType
from libcbm.storage.backends import numpy_backend


class PandasDataFrameBackend(DataFrame):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def __getitem__(self, col_name: str) -> Series:
        data = self._df[col_name]
        return PandasSeriesBackend(col_name, data)

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

    def assign(self, col_name: str, value: Any, indices: Series = None):
        if indices is not None:
            self._df.iloc[
                indices.to_numpy(), self._df.columns.get_loc(col_name)
            ] = value
        else:
            self._df.iloc[:, self._df.columns.get_loc(col_name)] = value

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

    def to_c_contiguous_numpy_array(self) -> np.ndarray:
        return np.ascontiguousarray(self._df)

    def to_pandas(self) -> pd.DataFrame:
        return self._df

    def zero(self):
        self._df.iloc[:] = 0

    def map(self, arg: Union[dict, Callable]) -> DataFrame:
        cols = list(self._df.columns)
        output = pd.DataFrame(
            index=self._df.index,
            columns=cols,
            data={col: self._df[col].map(arg) for col in cols},
        )
        return PandasDataFrameBackend(output)

    def evaluate_filter(self, expression: str) -> Series:
        return PandasSeriesBackend(None, self._df.eval(expression))


class PandasSeriesBackend(Series):
    """
    Series is a wrapper for one of several underlying storage types which
    presents a limited interface for internal usage by libcbm.
    """

    def __init__(self, name: str, series: pd.Series):
        self._name = name
        self._series = series

    @property
    def name(self) -> str:
        return self._name

    def filter(self, arg: "Series") -> "Series":
        """
        Return a new series of the elements
        corresponding to the true values in the specified arg
        """
        return PandasSeriesBackend(
            self._name, self._series[arg.to_numpy()].reset_index(drop=True)
        )

    def take(self, indices: "Series") -> "Series":
        """return the elements of this series at the specified indices
        (returns a copy)"""
        return PandasSeriesBackend(
            self._name,
            self._series.iloc[indices.to_numpy].reset_index(drop=True),
        )

    def as_type(self, type_name: str) -> "Series":
        return PandasSeriesBackend(self._name, self._series.astype(type_name))

    def assign(self, indices: "Series", value: Union["Series", Any]):
        if isinstance(value, Series):
            self._series.iloc[indices.to_numpy()] = value.to_numpy()
        else:
            self._series.iloc[indices.to_numpy()] = value

    def assign_all(self, value: Union["Series", Any]):
        """
        set all values in this series to the specified value
        """
        if isinstance(value, Series):
            self._series.iloc[:] = value.to_numpy()
        else:
            self._series.iloc[:] = value

    def map(self, arg: Union[dict, Callable[[int, Any], Any]]) -> "Series":
        return PandasSeriesBackend(self._name, self._series.map(arg))

    def at(self, idx: int) -> Any:
        """Gets the value at the specified sequential index"""
        return self._series.iloc[idx]

    def any(self) -> bool:
        """
        return True if at least one value in this series is
        non-zero
        """
        return self._series.any()

    def unique(self) -> "Series":
        return PandasSeriesBackend(
            self._name, pd.Series(name=self._name, data=self._series.unique())
        )

    def to_numpy(self) -> np.ndarray:
        return self._series.to_numpy()

    def to_numpy_ptr(self) -> ctypes.pointer:
        if str(self._series.dtype) == "int32":
            ptr_type = ctypes.c_int32
        elif str(self._series.dtype) == "float64":
            ptr_type = ctypes.c_double
        else:
            raise ValueError(
                f"series type not supported {str(self._series.dtype)}"
            )
        return numpy_backend.get_numpy_pointer(
            self._series.to_numpy(), ptr_type
        )

    def less(self, other: "Series") -> "Series":
        """
        returns a boolean series with:
            True - where this series is less than the other series
            False - where this series is greater than or equal to the other
                series
        """
        return PandasSeriesBackend(
            self._name, (self._series < other.to_numpy())
        )

    def sum(self) -> Union[int, float]:
        return self._series.sum()

    @property
    def length(self) -> int:
        return self._series.size

    @property
    def backend_type(self) -> BackendType:
        return BackendType.pandas

    def __mul__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (self._series * other).reset_index(drop=True)
        )

    def __rmul__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (other * self._series).reset_index(drop=True)
        )

    def __add__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (other + self._series).reset_index(drop=True)
        )

    def __radd__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (other + self._series).reset_index(drop=True)
        )

    def __ge__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (other >= self._series).reset_index(drop=True)
        )

    def __gt__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (other > self._series).reset_index(drop=True)
        )

    def __le__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (other <= self._series).reset_index(drop=True)
        )

    def __lt__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (other < self._series).reset_index(drop=True)
        )

    def __eq__(self, other: Union[int, float, "Series"]) -> "Series":
        return PandasSeriesBackend(
            self._name, (other == self._series).reset_index(drop=True)
        )


def concat_data_frame(
    dfs: list[PandasDataFrameBackend],
) -> PandasDataFrameBackend:
    return PandasDataFrameBackend(
        pd.concat([d._df for d in dfs], ignore_index=True)
    )


def concat_series(series: list[PandasSeriesBackend]) -> PandasSeriesBackend:
    pd.concat([s._series for s in series])


def logical_and(
    s1: PandasSeriesBackend, s2: PandasSeriesBackend
) -> PandasSeriesBackend:
    return PandasSeriesBackend(None, s1._series & s2._series)


def logical_not(series: PandasSeriesBackend) -> PandasSeriesBackend:
    return PandasSeriesBackend(None, ~(series._series))


def logical_or(
    s1: PandasSeriesBackend, s2: PandasSeriesBackend
) -> PandasSeriesBackend:
    return PandasSeriesBackend(None, s1._series | s2._series)


def make_boolean_series(init: bool, size: int) -> PandasSeriesBackend:
    return PandasSeriesBackend(
        None, pd.Series(np.full(shape=size, fill_value=init, dtype="bool"))
    )


def is_null(series: PandasSeriesBackend) -> PandasSeriesBackend:
    return PandasSeriesBackend(None, pd.Series(pd.isnull(series._series)))


def indices_nonzero(series: PandasSeriesBackend) -> PandasSeriesBackend:
    return PandasSeriesBackend(
        None, pd.Series(series._series.to_numpy().nonzero()[0])
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
        pd.DataFrame({s.name: s._series for s in series_list})
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
