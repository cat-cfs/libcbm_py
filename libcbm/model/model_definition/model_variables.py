from __future__ import annotations
from libcbm.storage.dataframe import DataFrame
from libcbm.storage import dataframe
import pandas as pd


class ModelVariables:
    """
    Container for multiple named dataframes.  Dataframes are assumed to be
    row-index-aligined
    """

    def __init__(self, data: dict[str, DataFrame]):
        self._data = data

    def __getitem__(self, name: str) -> DataFrame:
        return self._data[name]

    def __setitem__(self, name: str, new_value: DataFrame):
        self._data[name] = dataframe.convert_dataframe_backend(
            new_value, self._data[name].backend_type
        )

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def get_collection(self) -> dict[str, DataFrame]:
        """
        Get a dictionary containing named references to the
        dataframes stored in this collection
        """
        return {k: v for k, v in self._data.items()}

    @staticmethod
    def from_pandas(frames: dict[str, pd.DataFrame]) -> "ModelVariables":
        """
        Assemble a ModelVariables instance from a collection of pandas
        DataFrames
        """
        return ModelVariables(
            {k: dataframe.from_pandas(v) for k, v in frames.items()}
        )

    def to_pandas(self) -> dict[str, pd.DataFrame]:
        """
        return the dataframes in this collection as a dictionary
        of named pandas dataframes.  This may result in a copy
        if the underlying dataframe storage backend is not pandas
        """
        return {k: v.to_pandas() for k, v in self._data.items()}
