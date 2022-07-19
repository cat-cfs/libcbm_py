import pandas as pd
from unittest.mock import patch
from libcbm.model.cbm.cbm_variables import CBMVariables
from libcbm.model.cbm.cbm_output import CBMOutput
from libcbm.storage.backends import BackendType
from libcbm.storage.dataframe import from_pandas


def _make_test_data() -> CBMVariables:
    return CBMVariables(
        pools=from_pandas(pd.DataFrame({"p1": [1, 2, 3]})),
        flux=from_pandas(pd.DataFrame({"f1": [1, 2, 3]})),
        classifiers=from_pandas(
            pd.DataFrame({"c1": [1, 1, 1], "c2": [2, 2, 2]})
        ),
        state=from_pandas(pd.DataFrame({"s1": [1, 1, 1]})),
        inventory=from_pandas(
            pd.DataFrame({"i1": [1, 2, 3], "area": [1, 2, 3]})
        ),
        parameters=from_pandas(pd.DataFrame({"p1": [-1, -1, -1]})),
    )


@patch("libcbm.model.cbm.cbm_output.dataframe")
@patch("libcbm.model.cbm.cbm_output.series")
def test_construction(series, dataframe):
    cbm_output = CBMOutput(
        density=True,
        classifier_map={1: "a"},
        disturbance_type_map={2: "b"},
        backend_type=BackendType.pandas,
    )

    assert cbm_output.density == True
    assert cbm_output.disturbance_type_map == {2: "b"}
    assert cbm_output.classifier_map == {1: "a"}
    assert cbm_output.backend_type == BackendType.pandas
    assert cbm_output.pools is None
    assert cbm_output.flux is None
    assert cbm_output.state is None
    assert cbm_output.classifiers is None
    assert cbm_output.parameters is None
    assert cbm_output.area is None


def test_append_simulation_result_density_false():
    cbm_output = CBMOutput(
        density=False,
        classifier_map=None,
        disturbance_type_map=None,
        backend_type=BackendType.pandas,
    )

    cbm_output.append_simulation_result(timestep=1, cbm_vars=_make_test_data())
    assert cbm_output.pools.to_pandas().equals(
        pd.DataFrame(
            {"identifier": [1, 2, 3], "timestep": [1, 1, 1], "p1": [1, 4, 9]}
        )
    )
    assert cbm_output.flux.to_pandas().equals(
        pd.DataFrame(
            {"identifier": [1, 2, 3], "timestep": [1, 1, 1], "f1": [1, 4, 9]}
        )
    )


def test_append_simulation_result_no_mapping():
    cbm_output = CBMOutput(
        density=True,
        classifier_map=None,
        disturbance_type_map=None,
        backend_type=BackendType.pandas,
    )

    cbm_output.append_simulation_result(timestep=1, cbm_vars=_make_test_data())
    assert cbm_output.pools.to_pandas().equals(
        pd.DataFrame(
            {"identifier": [1, 2, 3], "timestep": [1, 1, 1], "p1": [1, 2, 3]}
        )
    )
    assert cbm_output.flux.to_pandas().equals(
        pd.DataFrame(
            {"identifier": [1, 2, 3], "timestep": [1, 1, 1], "f1": [1, 2, 3]}
        )
    )

    # flux=from_pandas(pd.DataFrame({"f1": [1, 2, 3]})),
    # classifiers=from_pandas(
    #    pd.DataFrame({"c1": [1, 1, 1], "c2": [2, 2, 2]})
    # ),
    # state=from_pandas(pd.DataFrame({"s1": [1, 1, 1]})),
    # inventory=from_pandas(
    #    pd.DataFrame({"i1": [1, 2, 3], "area": [1, 2, 3]})
    # ),
    # parameters=from_pandas(pd.DataFrame({"p1": [-1, -1, -1]})),

    # assert cbm_output.pools.to_pandas() = test_data.pools


def test_append_simulation_result_with_mapping():
    pass
