from enum import Enum
import pandas as pd
from typing import Union

from libcbm.storage.dataframe import DataFrame
from libcbm.storage.series import Series


class BackendType(Enum):
    numpy = 1
    pandas = 2
    pyarrow = 3


def get_backend(backend_type: BackendType):

    if backend_type == BackendType.numpy:
        from libcbm.storage.backends import numpy_backend

        return numpy_backend
    elif backend_type == BackendType.pandas:
        from libcbm.storage.backends import pandas_backend

        return pandas_backend
    else:
        raise NotImplementedError()


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
