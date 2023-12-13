import pytest
import numpy as np
import pandas as pd
from libcbm.model.model_definition.matrix_merge_index import MatrixMergeIndex


def test_error_on_non_integer_keys():
    # this should work
    MatrixMergeIndex(
        3,
        {
            "a": np.array([1, 2, 3], dtype="int8"),
        },
    )
    with pytest.raises(ValueError):
        # this wont work (floating point values in the key_data)
        MatrixMergeIndex(
            3,
            {
                "a": np.array([1, 2, 3], dtype="int8"),
                "b": np.array([1.1, 2.2, 3.3]),
            },
        )

    with pytest.raises(ValueError):
        # this wont work (string/objects in the key_data)
        df = pd.DataFrame({"a": ["a1", "b1", "c1"]})
        MatrixMergeIndex(1, {"a": df["a"].to_numpy()})


def test_error_on_not_found_keys_with_no_fill_value():
    m = MatrixMergeIndex(
        3,
        {
            "a": np.array([1, 2, 3], dtype="int8"),
        },
    )
    with pytest.raises(ValueError):
        # there is no 4 in the a key array above
        m.merge({"a": np.array([1, 2, 3, 4])})


def test_merge():
    m = MatrixMergeIndex(
        3,
        {
            "a": np.array([1, 2, 3], dtype="int8"),
            "b": np.array([1, 2, 3], dtype="int16"),
        },
    )
    result = m.merge(
        {"a": np.array([1, 1, 2, 3, 3, 3]), "b": np.array([1, 1, 2, 3, 3, 3])}
    )
    assert result.tolist() == [0, 0, 1, 2, 2, 2]


def test_merge_with_fill_value():
    m = MatrixMergeIndex(
        3,
        {
            "a": np.array([1, 2, 3], dtype="int64"),
            "b": np.array([1, 2, 3], dtype="int32"),
        },
    )
    result = m.merge(
        {
            "a": np.array([1, 1, 2, 3, 3, 3, 95]),
            "b": np.array([1, 1, 2, 3, 3, 3, 17]),
        },
        fill_value=1,
    )
    assert result.tolist() == [0, 0, 1, 2, 2, 2, 1]


def test_merge_error_fill_value_out_of_range():
    m = MatrixMergeIndex(
        3,
        {
            "a": np.array([1, 2, 3], dtype="int32"),
            "b": np.array([1, 2, 3], dtype="int64"),
        },
    )
    with pytest.raises(ValueError):
        m.merge({"a": np.array([1.0]), "b": np.array([1])}, fill_value=-1)

    with pytest.raises(ValueError):
        m.merge({"a": np.array([1.0]), "b": np.array([1])}, fill_value=3)
    with pytest.raises(ValueError):
        m.merge({"a": np.array([1.0]), "b": np.array([1])}, fill_value=1000)
