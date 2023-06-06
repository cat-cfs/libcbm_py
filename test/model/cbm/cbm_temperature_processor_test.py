import pytest
from types import SimpleNamespace
import pandas as pd
from libcbm.model.cbm.cbm_temperature_processor import (
    SpatialUnitMeanAnnualTemperatureProcessor,
)
from libcbm.storage import dataframe


def test_temperature_processor_error_no_t0_data():
    processor = SpatialUnitMeanAnnualTemperatureProcessor(
            pd.DataFrame(
                {
                "timestep": [1, 2, 1, 2],
                "spatial_unit": [1, 1, 2, 2],
                "mean_annual_temp": [0.2, 0.3, 0.4, 0.5],
                }
            )
        )
    with pytest.raises(ValueError):

        processor.get_spinup_parameters(
            inventory=dataframe.from_pandas(
                pd.DataFrame({"spatial_unit": [1, 1, 2, 2, 2]})
            ))
def test_temperature_processor_error_on_duplicate_key():
    with pytest.raises(ValueError):
        SpatialUnitMeanAnnualTemperatureProcessor(
            pd.DataFrame(
                {
                    "timestep": [0, 1, 2, 0, 1, 2],
                    "spatial_unit": [1, 1, 1, 1, 2, 2],
                    "mean_annual_temp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                }
            )
        )


def test_temperature_processor():
    processor = SpatialUnitMeanAnnualTemperatureProcessor(
        pd.DataFrame(
            {
                "timestep": [0, 1, 2, 0, 1, 2],
                "spatial_unit": [1, 1, 1, 2, 2, 2],
                "mean_annual_temp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )
    )

    spinup_parameters = processor.get_spinup_parameters(
        inventory=dataframe.from_pandas(
            pd.DataFrame({"spatial_unit": [1, 1, 2, 2, 2]})
        )
    ).to_pandas()

    assert spinup_parameters["mean_annual_temp"].to_list() == [
        0.1,
        0.1,
        0.4,
        0.4,
        0.4,
    ]

    mock_cbm_vars = SimpleNamespace(
        inventory=dataframe.from_pandas(
            pd.DataFrame({"spatial_unit": [1, 1, 2, 2, 2]})
        ),
        parameters=dataframe.from_pandas(pd.DataFrame()),
    )

    mock_cbm_vars = processor.set_timestep_mean_annual_temperature(
        1, mock_cbm_vars
    )
    assert mock_cbm_vars.parameters["mean_annual_temp"].to_list() == [
        0.2,
        0.2,
        0.5,
        0.5,
        0.5,
    ]

    mock_cbm_vars = processor.set_timestep_mean_annual_temperature(
        2, mock_cbm_vars
    )
    assert mock_cbm_vars.parameters["mean_annual_temp"].to_list() == [
        0.3,
        0.3,
        0.6,
        0.6,
        0.6,
    ]

    with pytest.raises(ValueError):
        # timestep 3 not defined
        processor.set_timestep_mean_annual_temperature(3, mock_cbm_vars)
