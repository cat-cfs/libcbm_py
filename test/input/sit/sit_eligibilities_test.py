import pytest
import pandas as pd
from libcbm.input.sit import sit_eligbility_parser


def test_parse_eligibilities():
    elgibilities_input = pd.DataFrame(
        columns=["id", "desc", "exp_type", "exp", "p1", "p2"],
        data=[
            [
                1,
                "filter desc",
                "state",
                "(a<{p1}) or (b=={p2})",
                "900",
                "-12",
            ],
            [
                1,
                "filter desc",
                "pool",
                "(c<{p1}) or (d=={p2})",
                "32",
                "-22",
            ],
            [
                2,
                "filter desc",
                "state",
                "",
                "9999",
                "8888",
            ],
            [
                3,
                "",
                "",
                "",
                "",
                "",
            ],
        ],
    )
    sit_eligibilities = sit_eligbility_parser.parse_eligibilities(
        elgibilities_input
    )
    assert sit_eligibilities["eligibility_id"].iloc[0] == 1
    assert (
        sit_eligibilities["pool_filter_expression"].iloc[0]
        == "((c<32.0) or (d==-22.0))"
    )
    assert (
        sit_eligibilities["state_filter_expression"].iloc[0]
        == "((a<900.0) or (b==-12.0))"
    )

    assert sit_eligibilities["eligibility_id"].iloc[1] == 2
    assert sit_eligibilities["pool_filter_expression"].iloc[1] == ""
    assert sit_eligibilities["state_filter_expression"].iloc[1] == ""

    assert sit_eligibilities["eligibility_id"].iloc[2] == 3
    assert sit_eligibilities["pool_filter_expression"].iloc[2] == ""
    assert sit_eligibilities["state_filter_expression"].iloc[2] == ""
    assert list(sit_eligibilities.columns) == [
        "eligibility_id",
        "pool_filter_expression",
        "state_filter_expression",
    ]


def test_error_on_missing_ids():
    event = pd.DataFrame(
        {
            "eligibility_id": [2],  # missing
        }
    )

    elgibilities = pd.DataFrame(
        columns=["eligibility_id", "pool_filter", "state_filter"],
        data=[[1, "", ""]],
    )

    with pytest.raises(ValueError):
        sit_eligbility_parser.validate_eligibilities_relationship(
            elgibilities, event
        )
