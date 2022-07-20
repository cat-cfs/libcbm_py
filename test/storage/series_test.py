import pytest
from libcbm.storage.backends import BackendType
from libcbm.storage import series
from libcbm.storage import dataframe


def test_series():
    s_base = series.from_list("series_name", list(range(0, 100)))
    for backend in BackendType:
        s = dataframe.convert_series_backend(s_base.copy(), backend)

        # name
        assert s.name == "series_name"
        s.name = f"{backend}_series"
        assert s.name == f"{backend}_series"

        # copy
        assert s.copy().to_list() == s_base.to_list()


        # filter
        assert s.filter(
            series.from_list("", [True] * 50 + [False] * 50)
        ).to_list() == list(range(0, 50))

        # take
        assert s.take(
            series.from_list("", [98, 3, 1, 2, 2, -1])
        ).to_list() == [98, 3, 1, 2, 2, 99]

        with pytest.raises(IndexError):
            # out of range
            s.take(series.from_list("", [100]))
        with pytest.raises(IndexError):
            # out of range
            s.take(series.from_list("", [-101]))

        # as_type
        assert (s.as_type("float") + 0.1).to_list() == [
            float(x) + 0.1 for x in range(0, 100)
        ]

        assert s.as_type("str").to_list() == [str(x) for x in range(0, 100)]

        # assign
        s_assigned = s.copy()
        s_assigned.assign(1)
        assert s_assigned.to_list() == [1] * 100
        s_assigned = s.copy()
        s_assigned.assign(
            series.from_list("", [99, 98, 97]), series.from_list("", [0, 1, 2])
        )
        assert s_assigned.to_list() == [99, 98, 97] + list(range(3, 100))

        # map
        s.map()
