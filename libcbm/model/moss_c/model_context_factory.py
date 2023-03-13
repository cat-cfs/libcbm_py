import os
import pandas as pd
from libcbm.model.moss_c.model_context import ModelContext
from libcbm.storage import dataframe
from libcbm.storage.dataframe import DataFrame
from libcbm.storage.backends import BackendType


class InputData:
    def __init__(
        self,
        decay_parameter: pd.DataFrame,
        disturbance_type: pd.DataFrame,
        disturbance_matrix: pd.DataFrame,
        moss_c_parameter: pd.DataFrame,
        inventory: pd.DataFrame,
        mean_annual_temperature: pd.DataFrame,
        merch_volume: pd.DataFrame,
        spinup_parameter: pd.DataFrame,
    ):
        self.decay_parameter = decay_parameter
        self.disturbance_type = disturbance_type
        self.disturbance_matrix = disturbance_matrix
        self.moss_c_parameter = moss_c_parameter
        self.inventory = inventory
        self.mean_annual_temperature = mean_annual_temperature
        self.merch_volume = merch_volume
        self.spinup_parameter = spinup_parameter


def _checked_merge(
    df1: pd.DataFrame, df2: pd.DataFrame, left_on: str
) -> pd.DataFrame:
    merged = df1.merge(
        df2, left_on=left_on, right_index=True, validate="m:1", how="left"
    )
    missing_values = merged[merged.isnull().any(axis=1)][left_on]
    if len(missing_values.index) > 0:
        raise ValueError(
            f"missing values for '{left_on}' "
            f"detected: {list(missing_values.unique()[0:10])}"
        )
    return merged


def _initialize_dynamics_parameter(input_data: InputData) -> DataFrame:
    max_vols = pd.DataFrame(
        {
            "max_merch_vol": input_data.merch_volume["volume"]
            .groupby(by=input_data.merch_volume.index)
            .max()
        }
    )

    dynamics_param = input_data.inventory
    dynamics_param = _checked_merge(
        dynamics_param,
        input_data.moss_c_parameter,
        left_on="moss_c_parameter_id",
    )
    dynamics_param = _checked_merge(
        dynamics_param,
        input_data.decay_parameter,
        left_on="decay_parameter_id",
    )
    dynamics_param = _checked_merge(
        dynamics_param,
        input_data.mean_annual_temperature,
        left_on="mean_annual_temperature_id",
    )
    dynamics_param = _checked_merge(
        dynamics_param,
        input_data.spinup_parameter,
        left_on="spinup_parameter_id",
    )
    dynamics_param = _checked_merge(
        dynamics_param, max_vols, left_on="merch_volume_id"
    )

    if (dynamics_param.index != input_data.inventory.index).any():
        raise ValueError()

    return dataframe.from_pandas(dynamics_param)


def create_from_csv(
    dir: str,
    decay_parameter_fn: str = "decay_parameter.csv",
    disturbance_type_fn: str = "disturbance_type.csv",
    disturbance_matrix_fn: str = "disturbance_matrix.csv",
    moss_c_parameter_fn: str = "moss_c_parameter.csv",
    inventory_fn: str = "inventory.csv",
    mean_annual_temperature_fn: str = "mean_annual_temperature.csv",
    merch_volume_fn: str = "merch_volume.csv",
    spinup_parameter_fn: str = "spinup_parameter.csv",
    backend_type: BackendType = BackendType.pandas,
) -> ModelContext:
    def read_csv(fn: str, index_col: str = "id"):
        path = os.path.join(dir, fn)
        return pd.read_csv(path, index_col=index_col)

    input_data = InputData(
        decay_parameter=read_csv(decay_parameter_fn),
        disturbance_type=read_csv(disturbance_type_fn),
        disturbance_matrix=read_csv(disturbance_matrix_fn),
        moss_c_parameter=read_csv(moss_c_parameter_fn),
        inventory=read_csv(inventory_fn),
        mean_annual_temperature=read_csv(mean_annual_temperature_fn),
        merch_volume=read_csv(merch_volume_fn),
        spinup_parameter=read_csv(spinup_parameter_fn),
    )
    parameters = _initialize_dynamics_parameter(input_data)
    return ModelContext(
        dataframe.from_pandas(input_data.inventory),
        parameters,
        dataframe.from_pandas(input_data.merch_volume),
        dataframe.from_pandas(input_data.disturbance_matrix),
        backend_type,
    )
