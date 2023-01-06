import pandas as pd
from pandas.testing import assert_frame_equal
from unittest.mock import patch
from libcbm.model.cbm.cbm_variables import CBMVariables
from libcbm.model.cbm.cbm_output import CBMOutput
from libcbm.storage.backends import BackendType
from libcbm.storage.dataframe import from_pandas


def _make_test_data() -> CBMVariables:
    return CBMVariables(
        pools=from_pandas(pd.DataFrame({"p1": [1.0, 2.0, 3.0]})),
        flux=from_pandas(pd.DataFrame({"f1": [1.0, 2.0, 3.0]})),
        classifiers=from_pandas(
            pd.DataFrame({"c1": [1, 1, 1], "c2": [2, 2, 2]})
        ),
        state=from_pandas(
            pd.DataFrame(
                {"s1": [1, 1, 1], "last_disturbance_type": [-1, 1, -1]}
            )
        ),
        inventory=from_pandas(
            pd.DataFrame({"i1": [1, 2, 3], "area": [1.0, 2.0, 3.0]})
        ),
        parameters=from_pandas(
            pd.DataFrame({"p1": [-1, -1, -1], "disturbance_type": [1, 2, -1]})
        ),
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

    assert cbm_output.density is True
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
        classifier_map={1: "a", 2: "b"},
        disturbance_type_map=None,
        backend_type=BackendType.pandas,
    )

    cbm_output.append_simulation_result(timestep=1, cbm_vars=_make_test_data())
    assert_frame_equal(
        cbm_output.pools.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1], dtype="int"),
                "p1": [1.0, 4.0, 9.0],
            }
        ),
    )

    assert_frame_equal(
        cbm_output.flux.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1], dtype="int"),
                "f1": [1.0, 4.0, 9.0],
            }
        ),
    )

    assert_frame_equal(
        cbm_output.classifiers.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1], dtype="int"),
                "c1": ["a", "a", "a"],
                "c2": ["b", "b", "b"],
            }
        ),
    )

    assert_frame_equal(
        cbm_output.state.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1], dtype="int"),
                "s1": [1, 1, 1],
                "last_disturbance_type": [-1, 1, -1],
            }
        ),
    )

    assert_frame_equal(
        cbm_output.area.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1], dtype="int"),
                "area": [1.0, 2.0, 3.0],
            }
        ),
    )

    assert_frame_equal(
        cbm_output.parameters.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1], dtype="int"),
                "p1": [-1, -1, -1],
                "disturbance_type": [1, 2, -1],
            }
        ),
    )


def test_append_simulation_result_no_mapping():
    cbm_output = CBMOutput(
        density=True,
        classifier_map=None,
        disturbance_type_map=None,
        backend_type=BackendType.pandas,
    )

    cbm_output.append_simulation_result(timestep=1, cbm_vars=_make_test_data())
    assert_frame_equal(
        cbm_output.pools.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1], dtype="int"),
                "p1": [1.0, 2.0, 3.0],
            }
        ),
    )
    assert_frame_equal(
        cbm_output.flux.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1], dtype="int"),
                "f1": [1.0, 2.0, 3.0],
            }
        ),
    )

    assert_frame_equal(
        cbm_output.classifiers.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1], dtype="int"),
                "c1": [1, 1, 1],
                "c2": [2, 2, 2],
            }
        ),
    )

    assert_frame_equal(
        cbm_output.state.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1], dtype="int"),
                "s1": [1, 1, 1],
                "last_disturbance_type": [-1, 1, -1],
            }
        ),
    )

    assert_frame_equal(
        cbm_output.area.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1], dtype="int"),
                "area": [1.0, 2.0, 3.0],
            }
        ),
    )

    assert_frame_equal(
        cbm_output.parameters.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1], dtype="int"),
                "p1": [-1, -1, -1],
                "disturbance_type": [1, 2, -1],
            }
        ),
    )


def test_append_simulation_result_with_mapping_multiple_append():
    cbm_output = CBMOutput(
        density=True,
        classifier_map={1: "c1", 2: "c2"},
        disturbance_type_map={-1: "-1", 0: "d0", 1: "d1", 2: "d2"},
        backend_type=BackendType.pandas,
    )
    cbm_output.append_simulation_result(timestep=1, cbm_vars=_make_test_data())
    cbm_output.append_simulation_result(timestep=2, cbm_vars=_make_test_data())
    assert_frame_equal(
        cbm_output.pools.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3, 1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1, 2, 2, 2], dtype="int"),
                "p1": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            }
        ),
    )
    assert_frame_equal(
        cbm_output.flux.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3, 1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1, 2, 2, 2], dtype="int"),
                "f1": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            }
        ),
    )

    assert_frame_equal(
        cbm_output.classifiers.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3, 1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1, 2, 2, 2], dtype="int32"),
                "c1": ["c1", "c1", "c1", "c1", "c1", "c1"],
                "c2": ["c2", "c2", "c2", "c2", "c2", "c2"],
            }
        ),
    )

    assert_frame_equal(
        cbm_output.state.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3, 1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1, 2, 2, 2], dtype="int32"),
                "s1": [1, 1, 1, 1, 1, 1],
                "last_disturbance_type": ["-1", "d1", "-1", "-1", "d1", "-1"],
            }
        ),
    )

    assert_frame_equal(
        cbm_output.area.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3, 1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1, 2, 2, 2], dtype="int32"),
                "area": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            }
        ),
    )

    assert_frame_equal(
        cbm_output.parameters.to_pandas(),
        pd.DataFrame(
            {
                "identifier": pd.Series([1, 2, 3, 1, 2, 3], dtype="int64"),
                "timestep": pd.Series([1, 1, 1, 2, 2, 2], dtype="int32"),
                "p1": [-1, -1, -1, -1, -1, -1],
                "disturbance_type": ["d1", "d2", "-1", "d1", "d2", "-1"],
            }
        ),
    )
