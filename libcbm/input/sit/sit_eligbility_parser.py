from __future__ import annotations
from typing import Union
import pandas as pd
from libcbm.input.sit import sit_parser
from libcbm.input.sit import sit_format


def _unpack_eligbility_preformat(preformat_df: pd.DataFrame) -> pd.DataFrame:
    row_values_by_id: dict[int, dict[str, Union[int, list]]] = {}
    for _, row in preformat_df.iterrows():
        row_id = int(row["eligibility_id"])
        pool_filter_expression = ""
        state_filter_expression = ""
        expression_type = row["expression_type"]
        parameter_cols = preformat_df.columns[4:]

        parameter_collection = {
            col: float(row[col])
            for col in parameter_cols
            if not (pd.isnull(row[col]) or str(row[col]).strip() == "")
        }

        def extract_formatted_expr() -> str:
            expr = str(row["expression"])
            if expr.strip() != "":
                return f"({expr.format(**parameter_collection)})"
            return ""

        if expression_type == "pool":
            pool_filter_expression = extract_formatted_expr()
        elif expression_type == "state":
            state_filter_expression = extract_formatted_expr()
        elif not pd.isnull(expression_type) and not expression_type == "":
            raise ValueError(f"uknown expression type {expression_type}")

        row_values = {
            "eligibility_id": row_id,
            "pool_filter_expression": pool_filter_expression,
            "state_filter_expression": state_filter_expression,
        }
        if row_id in row_values_by_id:
            matched_row = row_values_by_id[row_id]
            if row_values["pool_filter_expression"]:
                matched_row["pool_filter_expressions"].append(
                    row_values["pool_filter_expression"]
                )
            if row_values["state_filter_expression"]:
                matched_row["state_filter_expressions"].append(
                    row_values["state_filter_expression"]
                )
        else:
            row_values_by_id[row_id] = {
                "eligibility_id": row_id,
                "pool_filter_expressions": [
                    row_values["pool_filter_expression"]
                ]
                if row_values["pool_filter_expression"]
                else [],
                "state_filter_expressions": [
                    row_values["state_filter_expression"]
                ]
                if row_values["state_filter_expression"]
                else [],
            }
    out_data = []
    for v in row_values_by_id.values():
        out_data.append(
            {
                "eligibility_id": v["eligibility_id"],
                "pool_filter_expression": " & ".join(
                    v["pool_filter_expressions"]
                ),
                "state_filter_expression": " & ".join(
                    v["state_filter_expressions"]
                ),
            }
        )
    return pd.DataFrame(out_data)


def validate_eligibilities_relationship(
    eligibilities: pd.DataFrame,
    disturbance_events: pd.DataFrame = None,
    transition_rules: pd.DataFrame = None,
):
    """Checks that the eligibility values in sit_transitions and sit_events are
    all present in the specified eligibility table, raising an error if not.

    Args:
        eligibilities (pd.DataFrame): table of eligibilities
        disturbance_events (pd.DataFrame, optional): sit_events to check.
            Defaults to None.
        transition_rules (pd.DataFrame, optional): transition rules to check.
            Defaults to None.
    """

    if disturbance_events is not None:
        # confirm that each row in the disturbance events with an
        # eligibility id >= 0 has a corresponding record in the eligibilities
        # table
        missing_ids = set(disturbance_events["eligibility_id"]) - set(
            eligibilities["eligibility_id"]
        )
        if missing_ids:
            raise ValueError(
                "eligibility_id values found in sit_events "
                f"but not in sit_eligibilities {missing_ids}"
            )

    if transition_rules is not None:
        # confirm that each row in the disturbance events with an
        # eligibility id >= 0 has a corresponding record in the eligibilities
        # table
        missing_ids = set(transition_rules["eligibility_id"]) - set(
            eligibilities["eligibility_id"]
        )
        if missing_ids:
            raise ValueError(
                "eligibility_id values found in sit_transition rules "
                f"but not in sit_eligibilities {missing_ids}"
            )


def parse_eligibilities(sit_eligibilities: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and validate disturbance eligibilities which are a libcbm-specific
    alternative to the eligibility columns in the CBM-CFS3 sit_disturbance
    events input.

    The benefit of this format is that the number of columns in sit_events is
    greatly reduced, and arbitrary boolean expressions of stand pool and state
    values, rather than min/max ranges supported in the CBM3-SIT format may be
    used.

    2 expressions types are supported: pool and state.  pool expressions are
    in terms of any column that is defined in the cbm_vars.pools DataFrame and
    state, for any column defined in cbm_vars.state during CBM runtime.

    Example Input value:

    ==  ===============  ===============  =======================================   =====
    id  description      expression_type  expression                                p1
    ==  ===============  ===============  =======================================   =====
    1   min total merch  pool             (SoftwoodMerch + HardwoodMerch) >= {p1}   10
    2   min total merch  pool             (SoftwoodMerch + HardwoodMerch) >= {p1}   20
    2   age min          state            age > {p1}                                5.0
    2   age max          state            age < {p1}                                100.0
    3   NULL             NULL             NULL                                      0.0
    ==  ===============  ===============  =======================================   =====

    Example return value:

     ==   =====================================  =======================
     id   pool_filter_expression                 state_filter_expression
     ==   =====================================  =======================
     1    (SoftwoodMerch + HardwoodMerch) >= 10  NULL
     2    (SoftwoodMerch + HardwoodMerch) >= 20  (age > 5) & (age < 100)
     3    NULL                                   NULL
     ==   =====================================  =======================

    Return value notes:

    * The id field in the sit_eligibilities corresponds to sit events, or sit transition rules
    * expressions are parsed by the numexpr library
    * note brackets are required around nested boolean expressions
      joined by a boolean operator (eg &)
    * for both pool_filter_expression, and state_filter_expression,
      the expressions must evaluate to a True or False value.  False
      indicates that the stand records being evaluated for the
      corresponding disturbance event deemed ineligible for the
      disturbance. True indicates that the expressions does not
      eliminate the stand from eligibility.
    * for pool_filter_expression any CBM pool is acceptable.  The pool names
      are defined in the cbm_defaults database tables.
    * for state_filter_expression any of the state values may be used in the
      boolean expression. See:
      :py:func:`libcbm.model.cbm.cbm_variables.initialize_cbm_state_variables`

    The final eligibility is evaluated as follows:

     ====================== ======================= =================
     pool_filter_expression state_filter_expression deemed_ineligible
     ====================== ======================= =================
     NULL or TRUE           NULL or TRUE            FALSE
     NULL or TRUE           FALSE                   TRUE
     FALSE                  NULL or TRUE            TRUE
     FALSE                  FALSE                   TRUE
     ====================== ======================= =================

    Args:
        sit_eligibilities (pandas.DataFrame): table of id (int),
            state_filter expression (str), pool filter expression (str).
            The disturbance event eligibility_id column
            corresponds to the id column in this table.

    Raises:
        ValueError: at least one null id value was detected in the id column
            of the specified sit_eligibilities table.
        ValueError: duplicate id value was detected in the id column of the
            specified eligibilities table.

    Returns:
        pandas.DataFrame: the validated event eligibilities table
    """  # noqa E501
    eligibility_format = sit_format.get_eligibility_format(
        sit_eligibilities.shape[1]
    )

    eligibilities_preformat = sit_parser.unpack_table(
        sit_eligibilities,
        eligibility_format,
        "disturbance eligibilities",
    )

    eligibilities = _unpack_eligbility_preformat(eligibilities_preformat)

    if pd.isnull(eligibilities["eligibility_id"]).any():
        raise ValueError(
            "null values detected in eligibilities eligibility_id " "column"
        )
    if eligibilities["eligibility_id"].duplicated().any():
        raise ValueError(
            "duplicated eligibility_id values detected in " "eligibilities"
        )
    eligibilities = eligibilities.fillna("")
    return eligibilities
