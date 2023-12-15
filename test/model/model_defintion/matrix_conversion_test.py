import pandas as pd
from libcbm.model.model_definition import matrix_conversion


def test_filter_pools():
    result = matrix_conversion.filter_pools(
        {
            "A": 1,
            "B": 2,
        },
        op_data=pd.DataFrame(
            columns=[
                "A.A",
                "A.B",
                "A.C",
                "B.A",
                "B.B",
                "B.C",
                "C.A",
                "C.B",
                "C.C",
            ],
            data=[[1] * 9] * 5,
        ),
    )

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(columns=["A.A", "A.B", "B.A", "B.B"], data=[[1] * 4] * 5),
    )


def test_to_coo_matrix():
    result = matrix_conversion.to_coo_matrix(
        {
            "A": 0,
            "B": 1,
        },
        op_data=pd.DataFrame(
            columns=[
                "A.A",
                "A.B",
                "B.A",
                "B.B",
            ],
            data=[[1] * 4] * 3,
        ),
    )

    assert (
        result.toarray() == [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
        ]
    ).all()
