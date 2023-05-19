from __future__ import annotations
import numpy as np
import pandas as pd

from typing import Union
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
        """
        filter this dataframe with a boolean series of length n_rows.
        Rows corresponding to true in the specified series are returned
        as a new dataframe.
        """
        pass

    @abstractmethod  # pragma: no cover
    def take(self, indices: Series) -> "DataFrame":
        """
        Create a new dataframe where the new dataframe's rows are the
        row indicies in the indicies argument.
        """
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
        """
        return the number of rows in this dataframe
        """
        pass

    @property
    @abstractmethod  # pragma: no cover
    def n_cols(self) -> int:
        """
        return the number of columns in this dataframe
        """
        pass

    @property
    @abstractmethod  # pragma: no cover
    def columns(self) -> list[str]:
        """
        get the list of column names
        """
        pass

    @property
    @abstractmethod  # pragma: no cover
    def backend_type(self) -> BackendType:
        """
        get the backend storage type
        """
        pass

    @abstractmethod  # pragma: no cover
    def copy(self) -> "DataFrame":
        """
        produce a copy of this dataframe
        """
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
        """
        Add a column to the dataframe based on the provided named series.
        """
        pass

    @abstractmethod  # pragma: no cover
    def to_numpy(self, make_c_contiguous=True) -> np.ndarray:
        """
        Return a reference to the data stored in the dataframe
        as a numpy array if possible. Intended to be used with
        uniformly-typed, numeric dataframes only, and unintended
        effects, or errors may be raised if this is not the case.
        """
        pass

    @abstractmethod  # pragma: no cover
    def to_pandas(self) -> pd.DataFrame:
        """
        return the data in this dataframe as a pandas dataframe.
        """
        pass

    @abstractmethod  # pragma: no cover
    def zero(self):
        """
        Set all values in this dataframe to zero
        """
        pass

    @abstractmethod  # pragma: no cover
    def map(self, arg: dict) -> "DataFrame":
        """Apply the specified mapping dictionary on every element of this
        dataframe to project a new dataframe with updated values. The results
        has the same number of rows, columns and same column names.  If any
        value in this dataframe is not present in the supplied dictionary an
        error will be raised.
        """
        pass

    @abstractmethod  # pragma: no cover
    def evaluate_filter(self, expression: str) -> Series:
        """Use a filter expression to produce a true/false series

        Args:
            expression (str): the boolean expression in terms of
                dataframe column names and constant values

        Returns:
            Series: a boolean series
        """
        pass

    @abstractmethod  # pragma: no cover
    def sort_values(self, by: str, ascending: bool = True) -> "DataFrame":
        """Return a sorted version of this dataframe

        Args:
            by (str): a single column name to sort by
            ascending (bool, optional): sort ascending (when True) or
                descending (when false) by the column name. Defaults to True.

        Returns:
            DataFrame: a sorted dataframe
        """
        pass


def concat_data_frame(
    data: list[DataFrame], backend_type: BackendType = None
) -> DataFrame:
    """Concatenate dataframes along the row axis.

    Args:
        data (list[DataFrame]): dataframes to concatenate
        backend_type (BackendType, optional): backend storage type of the
            resulting dataframe. If unspecified the backend type of the
            first DataFrame in data is used. Defaults to None.

    Returns:
        DataFrame: concatenated dataframe
    """
    data = [d for d in data if d is not None]
    if not data:
        return None
    backend_type, uniform_dfs = get_uniform_backend(data, backend_type)
    return backends.get_backend(backend_type).concat_data_frame(uniform_dfs)


def concat_series(
    series: list[Series], backend_type: BackendType = None
) -> Series:
    """Concatenate series into a single series.

    Args:
        series (list[Series]): list of series to concatenate
        backend_type (BackendType, optional): backend storage type of the
            resulting series. If unspecified the backend type of the
            first Series in data is used. Defaults to None. Defaults to None.

    Returns:
        Series: concatenated series
    """
    backend_type, uniform_dfs = get_uniform_backend(series, backend_type)
    return backends.get_backend(backend_type).concat_series(uniform_dfs)


def logical_and(
    s1: Series, s2: Series, backend_type: BackendType = None
) -> Series:
    """take the elementwise logical and of 2 series

    Args:
        s1 (Series): arg1 of the logical and
        s2 (Series): arg2 of the logical and
        backend_type (BackendType, optional): backend type of the result.
            If not specified the backend type of the first arg is used.
            Defaults to None.

    Returns:
        Series: result
    """
    backend_type, uniform_dfs = get_uniform_backend([s1, s2], backend_type)
    return backends.get_backend(backend_type).logical_and(
        uniform_dfs[0], uniform_dfs[1]
    )


def logical_not(series: Series) -> Series:
    """Take the logical not of the specified series

    Args:
        series (Series): series to evaluate

    Returns:
        Series: result
    """
    return backends.get_backend(series.backend_type).logical_not(series)


def logical_or(
    s1: Series, s2: Series, backend_type: BackendType = None
) -> Series:
    """take the elementwise logical or of 2 series

    Args:
        s1 (Series): arg1 of the logical or
        s2 (Series): arg2 of the logical or
        backend_type (BackendType, optional): backend type of the result.
            If not specified the backend type of the first arg is used.
            Defaults to None.
    Returns:
        Series: result
    """
    backend_type, uniform_dfs = get_uniform_backend([s1, s2], backend_type)
    return backends.get_backend(backend_type).logical_or(
        uniform_dfs[0], uniform_dfs[1]
    )


def make_boolean_series(
    init: bool, size: int, backend_type: BackendType.numpy
) -> Series:
    """Make an initialized boolean series

    Args:
        init (bool): True or False, the entire Series is assigned this value.
        size (int): length of the resulting series
        backend_type (BackendType): backend type of the result.
            If not specified then BackendType.numpy is used.
            Defaults to None.

    Returns:
        Series: the boolean series
    """
    return backends.get_backend(backend_type).make_boolean_series(init, size)


def is_null(series: Series) -> Series:
    """returns an Series of True where any value in the specified series
    is null, None, or NaN, and otherwise False

    Args:
        series (Series): series to evaluate

    Returns:
        Series: boolean series
    """
    return backends.get_backend(series.backend_type).is_null(series)


def indices_nonzero(series: Series) -> Series:
    """return a series of the 0 based sequential index of non-zero values in
    the specfied series

    Args:
        series (Series): series to evaulate

    Returns:
        Series: indices of the non-zeros
    """
    return backends.get_backend(series.backend_type).indices_nonzero(series)


def numeric_dataframe(
    cols: list[str],
    nrows: int,
    back_end: BackendType,
    init: float = 0.0,
) -> DataFrame:
    """Make an initialized numeric-only dataframe

    Args:
        cols (list[str]): column names
        nrows (int): number of rows
        back_end (BackendType): backend storage type of the resulting dataframe
        init (float, optional): initialization value. Defaults to 0.0.

    Returns:
        DataFrame: initialized numeric dataframe
    """
    return backends.get_backend(back_end).numeric_dataframe(cols, nrows, init)


def from_series_list(
    data: list[Union[Series, SeriesDef]], nrows: int, back_end: BackendType
) -> DataFrame:
    """initialize a dataframe from a list of Series or SeriesDef objects

    Args:
        data (list[Union[Series, SeriesDef]]): series information
        nrows (int): number of rows, this parameter is used when SeriesDef
            object are provided, and otherwise ignored.
        back_end (BackendType): backend storage type of the resulting dataframe

    Returns:
        DataFrame: dataframe object
    """
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
    """Initialize a dataframe object from a dictionary of named Series or
    SeriesDef objects.  The dictionary keys are used as the columns in the
    resulting dataframe.

    Args:
        data (dict[str, Union[Series, SeriesDef]]): dictionary of named Series
            or SeriesDef objects.
        nrows (int): the number of rows
        back_end (BackendType): backend storage type of the resulting dataframe

    Returns:
        DataFrame: initialized dataframe
    """
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
    """Create a DataFrame object with a pandas dataframe

    Args:
        df (pd.DataFrame): a pandas dataframe

    Returns:
        DataFrame: a DataFrame instance
    """
    from libcbm.storage.backends import pandas_backend

    return pandas_backend.PandasDataFrameBackend(df)


def from_numpy(data: dict[str, np.ndarray]) -> DataFrame:
    """Create a DataFrame object from a dictionary of name, numpy array pairs.

    Each array is expected to be single dimension and of the same length.

    Args:
        data (dict[str, np.ndarray]): name array pairs

    Returns:
        DataFrame: the Dataframe instance
    """
    from libcbm.storage.backends import numpy_backend

    return numpy_backend.NumpyDataFrameFrameBackend(data)


def convert_series_backend(
    series: Series, backend_type: BackendType
) -> Series:
    """Change the backend storage type of an existing Series object.

    If the backend type of the specified series is the same as the
    specified backend type, the series is simply returned.

    Args:
        series (Series): a Series object
        backend_type (BackendType): the backend type

    Raises:
        NotImplementedError: the specified backend_type has not been
            implemented.

    Returns:
        Series: The converted series
    """
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
    """Change the backend storage type of an existing DataFrame object.

    If the backend type of the specified dataframe is the same as the
    specified backend type, the dataframe is simply returned.

    Args:
        df (DataFrame): a DataFrame object
        backend_type (BackendType): the backend type

    Raises:
        NotImplementedError: the specified backend_type has not been
            implemented.

    Returns:
        DataFrame: the converted DataFrame
    """
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
    """Convert the backend type of all specified dataframes, or series.
    Also used to assert backend type uniformity of collections of these
    objects.

    Args:
        data (list[Union[DataFrame, Series]]): list of DataFrame or Series
            objects to convert or on which assert uniformity.
        backend_type (BackendType, optional): the required backend type of
            the output.  This must be specified if the provided items in `data`
            are non-uniform. Defaults to None.

    Raises:
        ValueError: backend_type was not specified, and the provided dataframe
            or series objects were non-uniform.

    Returns:
        tuple[BackendType, list[Union[DataFrame, Series]]]: _description_
    """
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
