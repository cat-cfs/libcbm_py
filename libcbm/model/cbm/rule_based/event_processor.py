import numpy as np


def process_event(filter_evaluator, event_filter, undisturbed, target_func,
                  classifiers, inventory, pools, state_variables):
    """Computes a CBM rule based event by filtering and targeting a subset of
    the specified inventory.  In the case of merchantable or area targets
    splits may occur to meet a disturbance target exactly.

    Args:
        filter_evaluator (object): function to evaluate the specified
            event_filter parameter.
            See :py:mod:`libcbm.model.cbm.rule_based.rule_filter`
        event_filter (object): a filter object containing information to deem
            stands eligible or ineligible for events
            See :py:mod:`libcbm.model.cbm.rule_based.rule_filter`
        undisturbed (pandas.Series): a boolean value series indicating each
            specified index is eligible (True) or ineligible (False) for
            disturbance.
        target_func (func): a function for creating a disturbance target.
            See: :py:mod:`libcbm.model.cbm.rule_based.rule_target`
        classifiers (pandas.DataFrame): CBM classifier values
        inventory (pandas.DataFrame): CBM inventory
        pools (pandas.DataFrame): CBM simulation pools
        state_variables (pandas.DataFrame): CBM simulation state variables

    Returns:
        tuple: The computed disturbance index and pools, state variables,
            classifiers and inventory which can be modified when splitting
            occurs. See the return value of :py:func:`apply_rule_based_event`
    """

    filter_result = filter_evaluator(event_filter)

    # set to false those stands affected by a previous disturbance from
    # eligibility
    filter_result = np.logical_and(undisturbed, filter_result)

    target = target_func(
        pools, inventory, state_variables, filter_result)

    return apply_rule_based_event(
        target, classifiers, inventory, pools, state_variables)


def apply_rule_based_event(target, classifiers, inventory, pools,
                           state_variables):
    """Apply the specified target to the CBM simulation variables,
    splitting them if necessary.

    Args:
        target (pandas.DataFrame): dataframe describing the index of
            records to disturbance and area split proportions.  See return
            value of methods in
            :py:mod:`libcbm.model.cbm.rule_based.rule_target`
        classifiers (pandas.DataFrame): CBM classifier values
        inventory (pandas.DataFrame): CBM inventory
        pools (pandas.DataFrame): CBM simulation pools
        state_variables (pandas.DataFrame): CBM simulation state variables

    Returns:
        tuple: tuple of target information and copies of the 4 input variables
            with splits applied if necessary:

            1. target (dict): a dictionary describing the computed disturbacne
                target
            2. classifiers (pandas.DataFrame): CBM classifier values
            3. inventory (pandas.DataFrame): CBM inventory
            4. pools (pandas.DataFrame): CBM simulation pools
            5. state_variables (pandas.DataFrame): CBM simulation state
               variables

    """
    target_index = target["disturbed_index"]
    target_area_proportions = target["area_proportions"]

    updated_inventory = inventory.copy()
    updated_classifiers = classifiers.copy()
    updated_state_variables = state_variables.copy()
    updated_pools = pools.copy()

    splits = target_area_proportions < 1.0
    split_index = target_index[splits]
    split_inventory = updated_inventory.iloc[split_index].copy()
    if len(split_inventory.index) > 0:
        # reduce the area of the disturbed inventory by the disturbance area
        # proportion
        updated_inv_idx = updated_inventory.index[split_index]
        updated_inventory.loc[updated_inv_idx, "area"] = \
            updated_inventory.loc[updated_inv_idx, "area"] * \
            target_area_proportions[splits].to_numpy()

        # set the split inventory as the remaining undisturbed area
        split_inventory.area = \
            split_inventory.area * \
            (1.0 - target_area_proportions[splits].array)

        # create the updated inventory by appending the split records
        updated_inventory = updated_inventory.append(
            split_inventory).reset_index(drop=True)

        # Since classifiers, pools, and state variables are not altered here
        # (this is done in the model) splitting is just a matter of adding a
        # copy of the split values.
        updated_classifiers = updated_classifiers.append(
            updated_classifiers.iloc[split_index].copy()
            ).reset_index(drop=True)
        updated_state_variables = updated_state_variables.append(
            updated_state_variables.iloc[split_index].copy()
            ).reset_index(drop=True)
        updated_pools = updated_pools.append(
            updated_pools.iloc[split_index].copy()
            ).reset_index(drop=True)
    return (target, updated_classifiers, updated_inventory, updated_pools,
            updated_state_variables)
