# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations
import pandas as pd
from typing import Iterable
from libcbm.input.sit import sit_transition_rule_parser
from libcbm.model.cbm.rule_based.sit import sit_stand_filter
from libcbm.model.cbm.rule_based import rule_filter
from libcbm.model.cbm.rule_based.transition_rule_processor import (
    TransitionRuleProcessor,
)
from libcbm.model.cbm.rule_based.rule_filter import RuleFilter
from libcbm.storage.dataframe import DataFrame
from libcbm.storage import dataframe
from libcbm.model.cbm.cbm_variables import CBMVariables
from libcbm.model.cbm.rule_based.classifier_filter import ClassifierFilter


def create_split_proportions(
    tr_group_key: dict, tr_group: DataFrame, group_error_max: float
) -> list[float]:
    """Create proportions

    Args:
        tr_group_key (dict): the composite key common to the transition
            rule group members
        tr_group (DataFrame): The table of transition rule group
            members
        group_error_max (float): used as a threshold to test if the group's
            total percentage exceeds 100 percent.

    Raises:
        ValueError:  Thrown if the absolute difference of total percentage
            minus 100 percent is greater than the group_error_max threshold.

    Returns:
        list: a list of proportions whose sum is 1.0 for splitting records
            for transition
    """
    # dealing with a couple of cases here:
    #
    # if the sum of the percent column in the specified group is less then
    # 100% then the number of splits is len(tr_group)+1 since the remainder
    # is allowed and is modelled as "unchanged" as far as transitioning
    # classifiers, etc.

    percent_sum = tr_group["percent"].sum()
    if abs(percent_sum - 100) < group_error_max:
        return (tr_group["percent"] / percent_sum).to_list()
    elif percent_sum > 100:
        raise ValueError(
            f"total percent ({percent_sum}) in transition rule group "
            f"{tr_group_key} exceeds 100%"
        )
    else:
        remainder = 100 - percent_sum
        appended_percent = tr_group["percent"].to_list() + [remainder]
        appended_percent_sum = sum(appended_percent)
        return [p / appended_percent_sum for p in appended_percent]


def create_state_variable_filter(
    tr_group_key: dict, state_variables: DataFrame
) -> RuleFilter:
    """Create a filter based on transition rule state criteria for setting
    stands eligible or ineligible for transition.

    Args:
        tr_group_key (dict): dictionary of values common to a transition rule
            group
        state_variables (DataFrame): table of state values for the
            current simulation for which to create a filter.

    Returns:
        RuleFilter: a filter object
    """
    state_filter_expression = sit_stand_filter.create_state_filter_expression(
        tr_group_key, True
    )
    return rule_filter.create_filter(
        expression=state_filter_expression, data=state_variables
    )


def get_transition_rule_filters(
    classifier_filter: ClassifierFilter,
    tr_group_key: dict,
    cbm_vars: CBMVariables,
) -> list[RuleFilter]:
    """
    Build a filter for the default CBM SIT transtion rule format: transition
    rules are eligible, or ineligible based on the values of disturbance type,
    classifier set, and age ranges


    """
    dist_type_target = tr_group_key["disturbance_type_id"]
    classifier_set = [tr_group_key[x] for x in cbm_vars.classifiers.columns]
    tr_filters = [
        create_state_variable_filter(tr_group_key, cbm_vars.state),
        classifier_filter.create_classifiers_filter(
            classifier_set, cbm_vars.classifiers
        ),
        rule_filter.create_filter(
            expression=f"(disturbance_type == {dist_type_target})",
            data=cbm_vars.parameters,
        ),
    ]
    return tr_filters


def sit_transition_rule_iterator(
    sit_transitions: pd.DataFrame, classifier_names: list[str]
) -> Iterable[tuple[dict[str, str], pd.DataFrame]]:
    """Groups transition rules by classifiers, and eligibility criteria and
    yields the sequence of group_key, group.

    Args:
        sit_transitions (pandas.DataFrame): parsed sit_transitions. See
            :py:mod:`libcbm.input.sit.sit_transition_rule_parser`
        classifier_names (list): the list of classifier names which must
            correspond to the first len(classifier_names) columns of
            sit_transitions
    Raises:
        ValueError: the sum of the percent field for any grouped set of
            transition rule rows exceeded 100%

    Returns:
        Tuple:
            Item1: the "key values" of the grouped transition rule rows
            Item2: the rows which compose the transtion rule group, as a
            dataframe

    """
    if len(sit_transitions.index) == 0:
        return

    group_cols = classifier_names
    if "eligibility_id" in sit_transitions:
        group_cols.append("eligibility_id")
    else:
        group_cols.extend(
            [
                "min_age",
                "max_age",
                "disturbance_type_id",
            ]
        )
    if "spatial_reference" in sit_transitions:
        group_cols.extend(["spatial_reference"])

    # group transition rules by their filter criteria
    # (classifier set, age range, disturbance type)
    grouping = sit_transitions.groupby(group_cols)
    group_error_max = sit_transition_rule_parser.GROUPED_PERCENT_ERR_MAX
    for group_key, group in dict(list(grouping)).items():
        group_key_dict = dict(zip(group_cols, group_key))
        if group.percent.sum() > 100 + group_error_max:
            raise ValueError(
                "Greater than 100 percent sum for percent field in "
                f"grouped transition rules with: {group_key_dict}"
            )
        yield group_key_dict, group


class SITTransitionRuleProcessor:
    def __init__(
        self,
        transition_rule_processor: TransitionRuleProcessor,
        classifier_filter: ClassifierFilter,
        group_error_max: float,
    ):
        self._transition_rule_processor = transition_rule_processor
        self._classifier_filter = classifier_filter
        self._group_error_max = group_error_max

    def process_transition_rules(
        self,
        sit_transitions: pd.DataFrame,
        cbm_vars: CBMVariables,
        sit_eligibilities: pd.DataFrame = None,
    ) -> CBMVariables:
        """Process the specified SIT transition rules versus the current model
        state.

        Args:
            sit_transitions (pandas.DataFrame): sit formatted transition rules.
                See:
                :py:func:`libcbm.input.sit.sit_transition_rule_parser.parse`
            cbm_vars (CBMVariables): CBM model state.
            sit_eligibilities (pandas.DataFrame): table of eligibility
                expressions with foreign key "eligibility_id"

        Returns:
            CBMVariables: the input CBM model state with the transition rules
            applied.
        """
        if sit_transitions is None:
            return cbm_vars

        classifiers = cbm_vars.classifiers
        n_stands = classifiers.n_rows
        classifier_names = classifiers.columns
        transition_iterator = sit_transition_rule_iterator(
            sit_transitions, classifier_names
        )

        transition_mask = dataframe.make_boolean_series(
            False, n_stands, cbm_vars.classifiers.backend_type
        )

        eligibilty_expressions: dict[int, pd.Series] = None
        if (
            sit_eligibilities is not None
            and "eligibility_id" in sit_transitions
        ):
            eligibilty_expressions = {
                int(row["eligibility_id"]): row
                for _, row in sit_eligibilities.iterrows()
            }

        for tr_group_key, tr_group in transition_iterator:
            filters = self._create_filters(
                cbm_vars,
                tr_group_key,
                eligibilty_expressions,
            )

            split_proportions = create_split_proportions(
                tr_group_key, tr_group, self._group_error_max
            )

            (
                transition_mask,
                cbm_vars,
            ) = self._transition_rule_processor.apply_transition_rule(
                dataframe.from_pandas(tr_group),
                filters,
                split_proportions,
                transition_mask,
                cbm_vars,
            )
        return cbm_vars

    def _create_filters(
        self,
        cbm_vars: CBMVariables,
        tr_group_key: dict[str, str],
        eligibilty_expressions: dict[int, pd.Series] = None,
    ) -> list[RuleFilter]:
        if eligibilty_expressions:
            expression = eligibilty_expressions[
                int(tr_group_key["eligibility_id"])
            ]
            filters = [
                rule_filter.create_filter(
                    expression=expression["pool_filter_expression"],
                    data=cbm_vars.pools,
                ),
                rule_filter.create_filter(
                    expression=expression["state_filter_expression"],
                    data=cbm_vars.state,
                ),
                self._classifier_filter.create_classifiers_filter(
                    [tr_group_key[x] for x in cbm_vars.classifiers.columns],
                    cbm_vars.classifiers,
                ),
            ]
        else:
            filters = get_transition_rule_filters(
                self._classifier_filter, tr_group_key, cbm_vars
            )

        return filters
