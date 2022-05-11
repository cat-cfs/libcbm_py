from typing import Any
from typing import Union
from typing import Callable
import pandas as pd
import numpy as np

from libcbm.storage.dataframe import DataFrame
from libcbm.storage.series import Series
from libcbm.storage.backends import BackendType


class PandasDataFrameBackend(DataFrame):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def getitem(self, col_name: str) -> Series:
        data = self._df[col_name]
        return Series(col_name, self._df[col_name], str(data.dtype))

    def filter(self, arg: Series) -> DataFrame:
        return DataFrame(self._df[arg.to_numpy()], back_end=BackendType.pandas)

    def take(self, indices: Series) -> DataFrame:
        return DataFrame(
            self._df.iloc[indices.to_numpy()], back_end=BackendType.pandas
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

    def n_rows(self) -> int:
        return len(self._df.index)

    def n_cols(self) -> int:
        return len(self._df.columns)

    def columns(self) -> list[str]:
        return list(self._df.columns)

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
        return DataFrame(output, back_end=BackendType.pandas)
