# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import numpy as np
from libcbm.model.cbm.rule_based import rule_filter


class ProcessEventResult:
    """Storage class for the result of the :py:func:`process_event`
    function.

    Args:
        cbm_vars (object): an object containing dataframes that store cbm
            simulation state and variables
        filter_result (np.ndarray): numpy boolean array indicating for each
            stand index in cbm_vars the index was eligible for the event when
            true, and ineligible when false
        rule_target_result (RuleTargetResult): instance of
            :py:class:`libcbm.model.cbm.rule_based.rule_target.RuleTargetResult`
            indicating targeted stands for this event
    """
    def __init__(self, cbm_vars, filter_result, rule_target_result):

        self.cbm_vars = cbm_vars
        self.filter_result = filter_result
        self.rule_target_result = rule_target_result


def process_event(event_filters, undisturbed, target_func,
                  disturbance_type_id, cbm_vars):
    """Computes a CBM rule based event by filtering and targeting a subset of
    the specified inventory.  In the case of merchantable or area targets
    splits may occur to meet a disturbance target exactly.

    Args:
        event_filters (list): a list of filter objects containing information
            to deem stands eligible or ineligible for events
            See :py:mod:`libcbm.model.cbm.rule_based.rule_filter`
        undisturbed (pandas.Series): a boolean value series indicating each
            specified index is eligible (True) or ineligible (False) for
            disturbance.
        target_func (func): a function for creating a disturbance target.
            See: :py:mod:`libcbm.model.cbm.rule_based.rule_target`.
            The function return value is:
            :py:class:`libcbm.model.cbm.rule_based.rule_target.RuleTargetResult`
        disturbance_type_id (int): the id for the disturbance event being
            processed.
        cbm_vars (object): an object containing dataframes that store cbm
            simulation state and variables

    Returns:
        ProcessEventResult: instance of class containing results for the
            disturbance event
    """

    filter_result = rule_filter.evaluate_filters(*event_filters)

    # set to false those stands affected by a previous disturbance from
    # eligibility
    filter_result = np.logical_and(undisturbed, filter_result)

    rule_target_result = target_func(
        cbm_vars, filter_result)

    cbm_vars = apply_rule_based_event(
        rule_target_result.target, disturbance_type_id, cbm_vars)

    return ProcessEventResult(cbm_vars, filter_result, rule_target_result)


def apply_rule_based_event(target, disturbance_type_id, cbm_vars):
    """Apply the specified target to the CBM simulation variables,
    splitting them if necessary.

    Args:
        target (pandas.DataFrame): dataframe describing the index of
            records to disturbance and area split proportions.  See return
            value of methods in
            :py:mod:`libcbm.model.cbm.rule_based.rule_target`
        disturbance_type_id (int): the id for the disturbance event being
            applied.
        cbm_vars (object): an object containing dataframes that store cbm
            simulation state and variables

    Returns:
        object: updated and expanded cbm_vars

    """
    target_index = target["disturbed_index"]
    target_area_proportions = target["area_proportions"]

    splits = target_area_proportions < 1.0
    n_splits = (splits).sum()
    split_index = target_index[splits]
    split_inventory = cbm_vars.inventory.iloc[split_index].copy()

    # set the disturbance types for the disturbed indices, based on
    # the sit_event disturbance_type field.
    cbm_vars.parameters.disturbance_type[target_index] = disturbance_type_id

    if len(split_inventory.index) > 0:
        # reduce the area of the disturbed inventory by the disturbance area
        # proportion
        updated_inv_idx = cbm_vars.inventory.index[split_index]
        cbm_vars.inventory.loc[updated_inv_idx, "area"] = \
            cbm_vars.inventory.loc[updated_inv_idx, "area"] * \
            target_area_proportions[splits].to_numpy()

        # set the split inventory as the remaining undisturbed area
        split_inventory.area = \
            split_inventory.area * \
            (1.0 - target_area_proportions[splits].array)

        # create the updated inventory by appending the split records
        cbm_vars.inventory = cbm_vars.inventory.append(
            split_inventory).reset_index(drop=True)

        # Since classifiers, pools, flux, and state variables are not altered
        # here (this is done in the model) splitting is just a matter of
        # adding a copy of the split values.
        cbm_vars.classifiers = cbm_vars.classifiers.append(
            cbm_vars.classifiers.iloc[split_index].copy()
            ).reset_index(drop=True)
        cbm_vars.state = cbm_vars.state.append(
            cbm_vars.state.iloc[split_index].copy()
            ).reset_index(drop=True)
        cbm_vars.pools = cbm_vars.pools.append(
            cbm_vars.pools.iloc[split_index].copy()
            ).reset_index(drop=True)
        cbm_vars.flux = cbm_vars.flux.append(
            cbm_vars.flux.iloc[split_index].copy()
            ).reset_index(drop=True)

        new_params = cbm_vars.parameters.iloc[split_index].copy()
        new_params.disturbance_type = np.zeros(n_splits, dtype=np.int32)
        cbm_vars.parameters = cbm_vars.parameters.append(
            new_params).reset_index(drop=True)

    return cbm_vars
