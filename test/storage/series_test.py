import pytest
import numpy as np
from libcbm.storage.backends import BackendType
from libcbm.storage import series
from libcbm.storage import dataframe


def test_series():
    s_base = series.from_numpy("series_name", np.arange(0, 100, dtype="int32"))
    for backend in BackendType:
        s = dataframe.convert_series_backend(s_base.copy(), backend)

        assert s.backend_type == backend
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
        with pytest.raises(ValueError):
            series.from_list("", ["abc"]).as_type("int")

        # assign
        s_assigned = s.copy()
        s_assigned.assign(1)
        assert s_assigned.to_list() == [1] * 100
        s_assigned = s.copy()
        # assert that an empty indices array will result in no assignment
        s_assigned.assign(1, series.from_numpy("", np.array([], dtype="int")))
        assert s_assigned.to_list() == s.to_list()
        s_assigned = s.copy()
        s_assigned.assign(
            series.from_list("", [99, 98, 97]), series.from_list("", [0, 1, 2])
        )
        assert s_assigned.to_list() == [99, 98, 97] + list(range(3, 100))
        # confirm error is raised when type change is tried but not permitted
        with pytest.raises(ValueError):
            s_assigned.assign("A")

        # map
        s_mapped1 = s.map({x: x + 1 for x in range(0, 100)})
        assert s_mapped1.to_list() == [x + 1 for x in range(0, 100)]

        # map: if the specified map series is not empty, but the dict is empty,
        # confirm a value error is raised
        with pytest.raises(ValueError):
            s.map({})

        # map: if the map series is empty, just get an empty series back
        # whether or not the dict is empty
        empty_series_mapped = dataframe.convert_series_backend(
            series.from_list("name", []), backend
        )
        assert empty_series_mapped.map({}).length == 0
        # the type of the result is the same as the mapped series type
        assert (
            empty_series_mapped.map({}).to_numpy().dtype
            == empty_series_mapped.to_numpy().dtype
        )

        assert empty_series_mapped.map({"a": 1.9}).length == 0
        # the type of the result is inherited from the dictionary value type if
        # provided
        assert empty_series_mapped.map({"a": 1.9}).to_numpy().dtype == "float"

        with pytest.raises(KeyError):
            s.map({999: 2})

        # at
        assert s.at(0) == 0
        assert s.at(99) == 99

        assert s.any()
        assert not s.all()

        assert s.unique().to_list() == s.to_list()

        assert s.to_list() == list(range(0, 100))
        assert list(s.to_numpy()) == s.to_list()
        assert s.to_numpy_ptr()

        assert s.data is not None

        assert s.sum() == sum(range(0, 100))
        assert s.cumsum() == np.cumsum(range(0, 100))

        assert s.max() == 99
        assert s.min() == 0
        assert s.length == 100

        test_operands = [12.34, np.array([-999] * 100)]
        test_array = np.array(range(0, 100))
        test_operators = [
            "__mul__",
            "__rmul__",
            "__truediv__",
            "__rtruediv__",
            "__add__",
            "__radd__",
            "__sub__",
            "__rsub__",
            "__ge__",
            "__gt__",
            "__le__",
            "__lt__",
            "__eq__",
            "__ne__",
        ]
        with pytest.warns(RuntimeWarning):
            for op in test_operators:
                for operand in test_operands:
                    assert (
                        getattr(s, op)(operand).to_numpy()
                        == getattr(test_array, op)(operand)
                    ).all()

        bit_operators = [
            "__and__",
            "__or__",
            "__rand__",
            "__ror__",
        ]

        bit_operands = [
            True,
            False,
            np.array([True] * 100),
            np.array([False] * 100),
        ]

        for op in bit_operators:
            for operand in bit_operands:
                assert (
                    getattr(s > 50, op)(operand).to_numpy()
                    == getattr(test_array > 50, op)(operand)
                ).all()

        assert (~(s < 50) == (test_array >= 50)).all()

        assert not s_base.is_null().any()

        assert (
            s_base.indices_nonzero().to_list()
            == list(np.arange(0, 100, dtype="int32"))[1:]
        )
