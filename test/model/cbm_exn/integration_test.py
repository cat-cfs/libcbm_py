import tempfile
import pandas as pd
from libcbm.model.cbm_exn import cbm_exn_model
from libcbm.model.cbm_exn.parameters import parameter_extraction
from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm import resources


def test_cbm_exn_integration():

    with tempfile.TemporaryDirectory() as tempdir:
        parameter_extraction.extract(
            resources.get_cbm_defaults_path(), tempdir, locale_code="en-CA"
        )
        spinup_input = CBMVariables.from_pandas(
            {
                "parameters": pd.DataFrame(
                    {
                        "age": [10],
                        "area": [1],
                        "delay": [0],
                        "return_interval": [150],
                        "min_rotations": [10],
                        "max_rotations": [30],
                        "spatial_unit_id": [1],
                        "species": [1],
                        "mean_annual_temperature": [-1.0],
                        "historical_disturbance_type": [1],
                        "last_pass_disturbance_type": [1],
                    }
                ),
                "increments": pd.DataFrame(
                    {
                        "row_idx": [0, 0, 0, 0, 0, 0, 0],
                        "age": [1, 2, 3, 4, 5, 6, 7],
                        "merch_inc": [0.1] * 7,
                        "other_inc": [0.1] * 7,
                        "foliage_inc": [0.1] * 7,
                    }
                ),
            }
        )
        with cbm_exn_model.initialize(
            config_path=tempdir, pandas_interface=False
        ) as model:
            cbm_vars = model.spinup(spinup_input)
            cbm_vars = model.step(cbm_vars)
