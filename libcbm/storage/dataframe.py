import numpy as np
import pandas as pd

from typing import Union
from typing import Callable
from libcbm.storage.backends import BackendType
from libcbm.storage import backends
from libcbm.storage.series import Series
from libcbm.storage.series import SeriesDef
from abc import ABC
from abc import abstractmethod


class DataFrame(ABC):
    """
    DataFrame is a wrapper for one of several underlying storage types which
    presents a limited interface for internal usage by libcbm
    """

    @abstractmethod  # pragma: no cover
    def __getitem__(self, col_name: str) -> Series:
        pass

    @abstractmethod  # pragma: no cover
    def filter(self, arg: Series) -> "DataFrame":
        pass

    @abstractmethod  # pragma: no cover
    def take(self, indices: Series) -> "DataFrame":
        pass

    @abstractmethod  # pragma: no cover
    def at(self, index: int) -> dict:
        """
        get the row at the specified 0 based sequential index as a row
        dictionary
        """
        pass

    @property
    @abstractmethod  # pragma: no cover
    def n_rows(self) -> int:
        pass

    @property
    @abstractmethod  # pragma: no cover
    def n_cols(self) -> int:
        pass

    @property
    @abstractmethod  # pragma: no cover
    def columns(self) -> list[str]:
        pass

    @property
    @abstractmethod  # pragma: no cover
    def backend_type(self) -> BackendType:
        pass

    @abstractmethod  # pragma: no cover
    def copy(self) -> "DataFrame":
        """produce a new in-memory copy of this dataframe"""
        pass

    @abstractmethod  # pragma: no cover
    def multiply(self, series: Series) -> "DataFrame":
        """
        Multiply this dataframe elementwise by the specified series along the
        row axis. An error is raised if the series length is not the same as
        the number of rows in this dataframe.  Returns new DataFrame
        """
        pass

    @abstractmethod  # pragma: no cover
    def add_column(self, series: Series, index: int) -> None:
        pass

    @abstractmethod  # pragma: no cover
    def to_numpy(self, make_c_contiguous=True) -> np.ndarray:
        pass

    @abstractmethod  # pragma: no cover
    def to_pandas(self) -> pd.DataFrame:
        pass

    @abstractmethod  # pragma: no cover
    def zero(self):
        """
        Set all values in this dataframe to zero
        """
        pass

    @abstractmethod  # pragma: no cover
    def map(self, arg: Union[dict, Callable]) -> "DataFrame":
        """Apply the specified mapping arg on every element of this dataframe
        to project a new dataframe with updated values. The results has the
        same number of rows, columns and same column names
        """
        pass

    @abstractmethod  # pragma: no cover
    def evaluate_filter(self, expression: str) -> Series:
        pass

    @abstractmethod  # pragma: no cover
    def sort_values(self, by: str, ascending: bool = True) -> "DataFrame":
        pass


def concat_data_frame(
    data: list[DataFrame], backend_type: BackendType = None
) -> DataFrame:
    data = [d for d in data if d is not None]
    if not data:
        return None
    backend_type, uniform_dfs = get_uniform_backend(data, backend_type)
    return backends.get_backend(backend_type).concat_data_frame(uniform_dfs)


def concat_series(
    series: list[Series], backend_type: BackendType = None
) -> Series:
    backend_type, uniform_dfs = get_uniform_backend(series, backend_type)
    return backends.get_backend(backend_type).concat_series(uniform_dfs)


def logical_and(
    s1: Series, s2: Series, backend_type: BackendType = None
) -> Series:
    backend_type, uniform_dfs = get_uniform_backend([s1, s2], backend_type)
    return backends.get_backend(backend_type).logical_and(
        uniform_dfs[0], uniform_dfs[1]
    )


def logical_not(series: Series) -> Series:
    return backends.get_backend(series.backend_type).logical_not(series)


def logical_or(
    s1: Series, s2: Series, backend_type: BackendType = None
) -> Series:
    backend_type, uniform_dfs = get_uniform_backend([s1, s2], backend_type)
    return backends.get_backend(backend_type).logical_or(
        uniform_dfs[0], uniform_dfs[1]
    )


def make_boolean_series(
    init: bool, size: int, backend_type: BackendType.numpy
) -> Series:
    return backends.get_backend(backend_type).make_boolean_series(init, size)


def is_null(series: Series) -> Series:
    """
    returns an Series of True where any value in the specified series
    is null, None, or NaN, and otherwise False
    """
    return backends.get_backend(series.backend_type).is_null(series)


def indices_nonzero(series: Series) -> Series:
    """
    return a series of the 0 based sequential index of non-zero values in
    the specfied series
    """
    return backends.get_backend(series.backend_type).indices_nonzero(series)


def numeric_dataframe(
    cols: list[str],
    nrows: int,
    back_end: BackendType,
    init: float = 0.0,
) -> DataFrame:
    return backends.get_backend(back_end).numeric_dataframe(cols, nrows, init)


def from_series_list(
    data: list[Union[Series, SeriesDef]], nrows: int, back_end: BackendType
) -> DataFrame:

    data_series = []
    for s in data:
        if isinstance(s, Series):
            data_series.append(s)
        else:
            data_series.append(s.make_series(nrows, back_end))
    backend_type, uniform_series = get_uniform_backend(data_series, back_end)
    return backends.get_backend(backend_type).from_series_list(uniform_series)


def from_series_dict(
    data: dict[str, Union[Series, SeriesDef]],
    nrows: int,
    back_end: BackendType,
) -> DataFrame:
    data_series = []
    names = []
    for k, v in data.items():
        names.append(k)
        if isinstance(v, Series):
            data_series.append(v)
        else:
            data_series.append(v.make_series(nrows, back_end))
    backend_type, uniform_series = get_uniform_backend(data_series, back_end)
    return backends.get_backend(backend_type).from_series_dict(
        dict(zip(names, uniform_series))
    )


def from_pandas(df: pd.DataFrame) -> DataFrame:
    from libcbm.storage.backends import pandas_backend

    return pandas_backend.PandasDataFrameBackend(df)


def from_numpy(data: dict[str, np.ndarray]) -> DataFrame:
    from libcbm.storage.backends import numpy_backend

    return numpy_backend.NumpyDataFrameFrameBackend(data)


def convert_series_backend(
    series: Series, backend_type: BackendType
) -> Series:
    if series.backend_type == backend_type:
        return series

    elif backend_type == BackendType.numpy:
        from libcbm.storage.backends import numpy_backend

        return numpy_backend.NumpySeriesBackend(series.name, series.to_numpy())
    elif backend_type == BackendType.pandas:
        from libcbm.storage.backends import pandas_backend

        return pandas_backend.PandasSeriesBackend(
            series.name, pd.Series(series.to_numpy())
        )
    else:
        raise NotImplementedError()


def convert_dataframe_backend(
    df: DataFrame, backend_type: BackendType
) -> DataFrame:
    if df.backend_type == backend_type:
        return df
    elif backend_type == BackendType.numpy:
        from libcbm.storage.backends import numpy_backend

        return numpy_backend.NumpyDataFrameFrameBackend(
            {col: df[col].to_numpy() for col in df.columns}
        )
    elif backend_type == BackendType.pandas:
        from libcbm.storage.backends import pandas_backend

        return pandas_backend.PandasDataFrameBackend(
            pd.DataFrame({col: df[col].to_numpy() for col in df.columns})
        )
    else:
        raise NotImplementedError()


def get_uniform_backend(
    data: list[Union[DataFrame, Series]], backend_type: BackendType = None
) -> tuple[BackendType, list[Union[DataFrame, Series]]]:
    if backend_type is None:
        inferred_backend = None
        for _d in data:
            if inferred_backend is None:
                inferred_backend = _d.backend_type
            elif inferred_backend != _d.backend_type:
                raise ValueError(
                    "backend type must be specified with non-uniform list of "
                    "dataframes backends"
                )

        backend_type = inferred_backend
    output = []
    for _d in data:
        if isinstance(_d, DataFrame):
            output.append(convert_dataframe_backend(_d, backend_type))
        else:
            output.append(convert_series_backend(_d, backend_type))
    return backend_type, output
