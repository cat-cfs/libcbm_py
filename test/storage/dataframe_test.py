from libcbm.storage import dataframe
from libcbm.storage import series
import pandas as pd
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


def test_dataframe_uniform_matrix():
    pass
