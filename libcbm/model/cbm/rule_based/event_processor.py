# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import Callable
from libcbm.model.cbm.rule_based import rule_filter
from libcbm.model.cbm.rule_based.rule_filter import RuleFilter
from libcbm.model.cbm.rule_based.rule_target import RuleTargetResult
from libcbm.model.cbm.cbm_variables import CBMVariables
from libcbm.storage.dataframe import Series
from libcbm.storage.dataframe import DataFrame
from libcbm.storage import dataframe_functions


class ProcessEventResult:
    """Storage class for the result of the :py:func:`process_event`
    function.

    Args:
        cbm_vars (object): an object containing dataframes that store cbm
            simulation state and variables
        filter_result (Series): boolean array indicating for each
            stand index in cbm_vars the index was eligible for the event when
            true, and ineligible when false
        rule_target_result (RuleTargetResult): instance of
            :py:class:`libcbm.model.cbm.rule_based.rule_target.RuleTargetResult`
            indicating targeted stands for this event
    """

    def __init__(
        self,
        cbm_vars: CBMVariables,
        filter_result: Series,
        rule_target_result: RuleTargetResult,
    ):

        self.cbm_vars = cbm_vars
        self.filter_result = filter_result
        self.rule_target_result = rule_target_result


def process_event(
    event_filters: list[RuleFilter],
    undisturbed: Series,
    target_func: Callable[[CBMVariables, Series], RuleTargetResult],
    disturbance_type_id: int,
    cbm_vars: CBMVariables,
) -> ProcessEventResult:
    """Computes a CBM rule based event by filtering and targeting a subset of
    the specified inventory.  In the case of merchantable or area targets
    splits may occur to meet a disturbance target exactly.

    Args:
        event_filters (list): a list of filter objects containing information
            to deem stands eligible or ineligible for events
        undisturbed (Series): a boolean value series indicating each
            specified index is eligible (True) or ineligible (False) for
            disturbance.
        target_func (func): a function for creating a disturbance target.
        disturbance_type_id (int): the id for the disturbance event being
            processed.
        cbm_vars (CBMVariables): an object containing dataframes that store cbm
            simulation state and variables

    Returns:
        ProcessEventResult: instance of class containing results for the
            disturbance event
    """

    filter_result = rule_filter.evaluate_filters(*event_filters)

    # set to false those stands affected by a previous disturbance from
    # eligibility
    filter_result = dataframe_functions.logical_and(undisturbed, filter_result)

    rule_target_result = target_func(cbm_vars, filter_result)

    cbm_vars = apply_rule_based_event(
        rule_target_result.target, disturbance_type_id, cbm_vars
    )

    return ProcessEventResult(cbm_vars, filter_result, rule_target_result)


def apply_rule_based_event(
    target: DataFrame, disturbance_type_id: int, cbm_vars: CBMVariables
) -> CBMVariables:
    """Apply the specified target to the CBM simulation variables,
    splitting them if necessary.

    Args:
        target (DataFrame): object describing the index of
            records to disturbance and area split proportions.
        disturbance_type_id (int): the id for the disturbance event being
            applied.
        cbm_vars (CBMVariables): an object containing dataframes that store cbm
            simulation state and variables

    Returns:
        CBMVariables: updated and expanded cbm_vars

    """
    target_index = target["disturbed_index"]
    target_area_proportions = target["area_proportions"]

    splits = target_area_proportions < 1.0
    split_index = target_index[splits]
    split_inventory = cbm_vars.inventory.take(split_index)

    # set the disturbance types for the disturbed indices, based on
    # the sit_event disturbance_type field.
    cbm_vars.parameters["disturbance_type"].assign(
        target_index, disturbance_type_id
    )

    if split_inventory.n_rows > 0:
        # reduce the area of the disturbed inventory by the disturbance area
        # proportion
        cbm_vars.inventory.assign(
            "area",
            (
                cbm_vars.inventory["area"].take(split_index)
                * target_area_proportions[splits]
            ),
            split_index,
        )

        # set the split inventory as the remaining undisturbed area
        split_inventory.assign(
            "area",
            split_inventory["area"] * (1.0 - target_area_proportions[splits]),
        )

        # create the updated inventory by appending the split records
        cbm_vars.inventory = dataframe_functions.concat_data_frame(
            [cbm_vars.inventory, split_inventory]
        )

        # Since classifiers, pools, flux, and state variables are not altered
        # here (this is done in the model) splitting is just a matter of
        # adding a copy of the split values.
        cbm_vars.classifiers = dataframe_functions.concat_data_frame(
            [cbm_vars.classifiers, cbm_vars.classifiers.take(split_index)]
        )
        cbm_vars.state = dataframe_functions.concat_data_frame(
            [cbm_vars.state, cbm_vars.state.take(split_index)]
        )
        cbm_vars.pools = dataframe_functions.concat_data_frame(
            [cbm_vars.pools, cbm_vars.pools.take(split_index)]
        )
        cbm_vars.flux = dataframe_functions.concat_data_frame(
            [cbm_vars.flux, cbm_vars.flux.take(split_index)]
        )

        new_params = cbm_vars.parameters.take(split_index)
        new_params.assign("disturbance_type", 0)
        cbm_vars.parameters = dataframe_functions.concat_data_frame(
            [cbm_vars.parameters, new_params]
        )

    return cbm_vars
