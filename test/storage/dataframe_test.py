import pytest
import numpy as np
import pandas as pd
from libcbm.storage import dataframe
from libcbm.storage import series
from libcbm.storage.backends import BackendType


def test_dataframe_mixed_types():
    test_data = dataframe.from_pandas(
        pd.DataFrame(
            data={
                "A": [1, 2, 3],
                "B": [1.1, 2.2, 3.3],
                "C": ["c1", "c2", "c3"],
            }
        )
    )

    for backend_type in BackendType:
        data = dataframe.convert_dataframe_backend(
            test_data.copy(), backend_type
        )
        assert data["A"].to_list() == [1, 2, 3]
        assert data["B"].to_list() == [1.1, 2.2, 3.3]
        assert data["C"].to_list() == ["c1", "c2", "c3"]

        filtered = data.filter(series.from_list("", [True, False, True]))
        assert filtered.n_rows == 2
        assert filtered.at(0) == {"A": 1, "B": 1.1, "C": "c1"}
        assert filtered.at(1) == {"A": 3, "B": 3.3, "C": "c3"}

        with pytest.raises(IndexError):
            data.take(series.from_list("", [100]))
        taken = data.take(series.from_list("", [0, 2, 0, 0]))
        assert taken.n_rows == 4
        for i in [0, 2, 3]:
            assert taken.at(i) == {"A": 1, "B": 1.1, "C": "c1"}
        assert taken.at(1) == {"A": 3, "B": 3.3, "C": "c3"}

        assert data.n_rows == 3
        assert data.n_cols == 3
        assert data.columns == ["A", "B", "C"]
        assert data.backend_type == backend_type
        data_copy = data.copy()
        assert data_copy.to_pandas().equals(data.to_pandas())
        data_copy.add_column(series.from_list("new_series", [1, 2, 3]), 1)
        assert data_copy["new_series"].to_list() == [1, 2, 3]
        assert data_copy.columns == ["A", "new_series", "B", "C"]
        assert data.to_pandas().equals(test_data.to_pandas())
        data_copy.zero()
        assert data_copy["new_series"].to_list() == [0, 0, 0]

        map_data = dataframe.from_pandas(
            pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        )
        mapped_data = map_data.map({x: 6 - x for x in range(0, 7)})
        assert mapped_data.to_pandas().equals(
            pd.DataFrame({"a": [5, 4, 3], "b": [2, 1, 0]})
        )
        with pytest.raises(KeyError):
            map_data.map({0: 0})

        filter_series = data.evaluate_filter("B > 2")
        assert filter_series.to_list() == [False, True, True]

        assert (
            data.sort_values(by="A", ascending=False)
            .to_pandas()
            .reset_index(drop=True)
            .equals(
                pd.DataFrame(
                    data={
                        "A": [3, 2, 1],
                        "B": [3.3, 2.2, 1.1],
                        "C": ["c3", "c2", "c1"],
                    }
                )
            )
        )


def test_dataframe_uniform_matrix():

    for backend_type in BackendType:
        data = dataframe.numeric_dataframe(
            cols=["A", "B", "C"], nrows=3, back_end=backend_type, init=2.0
        )

        for c in ["A", "B", "C"]:
            assert data[c].to_list() == [2.0, 2.0, 2.0]

        filtered = data.filter(series.from_list("", [True, False, True]))
        assert filtered.n_rows == 2
        assert (filtered.to_numpy() == np.full((2, 3), 2.0)).all()
        with pytest.raises(IndexError):
            data.take(series.from_list("", [100]))

        assert (
            data.take(series.from_list("", [0, 1, 2, 0, 1, 2])).to_numpy()
            == np.full((6, 3), 2.0)
        ).all()

        assert data.n_rows == 3
        assert data.n_cols == 3
        assert data.columns == ["A", "B", "C"]
        assert data.backend_type == backend_type

        data_copy = data.copy()
        assert data_copy.to_pandas().equals(data.to_pandas())

        assert (data.map({2.0: 10.0}).to_numpy() == np.full((3,3), 10.0)).all()

        with pytest.raises(KeyError):
            data.map({0: 0})
