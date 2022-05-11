from typing import Any
from typing import Union
from typing import Callable
import pandas as pd
import numpy as np

from libcbm.storage.dataframe import DataFrame
from libcbm.storage.series import Series
from libcbm.storage.backends import BackendType


def getitem(df, col_name: str) -> Series:
    data = df[col_name]
    return Series(col_name, df[col_name], str(data.dtype))


def filter(df: pd.DataFrame, arg: Series) -> DataFrame:
    return DataFrame(df[arg.to_numpy()], back_end=BackendType.pandas)


def take(df: pd.DataFrame, indices: Series) -> DataFrame:
    return DataFrame(df.iloc[indices.to_numpy()], back_end=BackendType.pandas)


def at(df: pd.DataFrame, index: int) -> dict:
    return df.iloc[index].to_dict()


def assign(
    df: pd.DataFrame, col_name: str, value: Any, indices: Series = None
):
    if indices is not None:
        df.iloc[indices.to_numpy(), df.columns.get_loc(col_name)] = value
    else:
        df.iloc[:, df.columns.get_loc(col_name)] = value


def n_rows(df: pd.DataFrame) -> int:
    return len(df.index)


def n_cols(df: pd.DataFrame) -> int:
    return len(df.columns)


def columns(df: pd.DataFrame) -> list[str]:
    return list(df.columns)


def backend_type() -> BackendType:
    return BackendType.pandas


def copy(df: pd.DataFrame) -> DataFrame:
    return DataFrame(df.copy(), back_end=BackendType.pandas)


def multiply(df: pd.DataFrame, series: Series) -> DataFrame:
    result = df.multiply(series.to_numpy(), axis=0)
    return DataFrame(result, back_end=BackendType.pandas)


def add_column(df: pd.DataFrame, series: Series, index: int) -> None:
    df.insert(index, series.name, series.to_numpy())


def to_c_contiguous_numpy_array(df: pd.DataFrame) -> np.ndarray:
    return np.ascontiguousarray(df)


def to_pandas(df: pd.DataFrame) -> pd.DataFrame:
    return df


def zero(df: pd.DataFrame):
    df.iloc[:] = 0


def map(df: pd.DataFrame, arg: Union[dict, Callable]) -> DataFrame:
    cols = list(df.columns)
    output = pd.DataFrame(
        index=df.index,
        columns=cols,
        data={col: df[col].map(arg) for col in cols},
    )
    return DataFrame(output, back_end=BackendType.pandas)


def numeric_dataframe(
    cols: list[str], nrows: int, init: float = 0.0
) -> DataFrame:
    df = pd.DataFrame(
        columns=cols, data=np.full(shape=(nrows, len(cols)), fill_value=init)
    )
    return DataFrame(df, back_end=BackendType.pandas)


def from_pandas(df: pd.DataFrame) -> DataFrame:
    return DataFrame(df, back_end=BackendType.pandas)


def from_series_dict(
    series_dict: dict[str, Series], back_end: BackendType, nrows: int
) -> DataFrame:
    return


def from_series_list(
    series_list: list[Series], back_end: BackendType, nrows: int
) -> DataFrame:
    return
