from libcbm.storage.dataframe import DataFrame
from libcbm.storage import dataframe
import pandas as pd


class CBMVariables:
    def __init__(self, data: dict[str, DataFrame]):
        self._data = data

    def __getitem__(self, name: str) -> DataFrame:
        return self._data[name]

    def get_collection(self) -> dict[str, DataFrame]:
        return {k: v for k, v in self._data.items()}

    @staticmethod
    def from_pandas(frames: dict[str, pd.DataFrame]) -> "CBMVariables":
        return CBMVariables(
            {k: dataframe.from_pandas(v) for k, v in frames.items()}
        )

    def to_pandas(self) -> dict[str, pd.DataFrame]:
        return {k: v.to_pandas() for k, v in self._data.items()}
