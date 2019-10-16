import numpy as np


def process_event(filter_evaluator, event_filter, undisturbed, target_func,
                  classifiers, inventory, pools, state_variables):
    """Computes a CBM rule based event by filtering and targeting a subset of
    the specified inventory.  In the case of merchantable or area targets
    splits may occur to meet a disturbance target exactly.

    Args:
        filter_factory (object): collection of methods to create and apply
            filters for CBM stands.
            See :py:mod:`libcbm.model.cbm.rule_based.rule_filter`
        classifiers_filter_factory (object): collection of methods to filter
            stands by classifier values using classifier sets.
            See:
            :py:class:`libcbm.model.cbm.rule_based.classifier_filter.ClassifierFilter`
        filter_data (object): object containing pool and state variable filter
            expressions. See:
            :py:func:`libcbm.model.cbm.rule_based.rule_filter.create_filter`
            for creation methods. Has the following properties:

            - pool_filter_expression (str): a boolean formatted expression in
                terms of CBM pools
            - pool_filter_columns (list): list of the string names of pools
                referenced in the expression.
            - state_filter_expression (str): a boolean formatted expression in
                terms of CBM state variables
            - state_filter_columns (list): list of the string names of state
                variables referenced in the expression.
            - classifier_set (list): a classifier set, which is a list of
                classifier values, aggregates or wildcards used to filter
                stands for eligibility.

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

    filtered_inventory = inventory[filter_result]
    filtered_state_variables = state_variables[filter_result]
    filtered_pools = pools[filter_result]

    target = target_func(
        filtered_pools, filtered_inventory, filtered_state_variables)

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
        tuple: tuple of copies of the 4 input variables with splits applied if
            necessary:

            1. classifiers (pandas.DataFrame): CBM classifier values
            2. inventory (pandas.DataFrame): CBM inventory
            3. pools (pandas.DataFrame): CBM simulation pools
            4. state_variables (pandas.DataFrame): CBM simulation state
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
        updated_inventory.area[split_index] = \
            updated_inventory.area[split_index] * \
            target_area_proportions[splits].array

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
    return (updated_classifiers, updated_inventory, updated_pools,
            updated_state_variables)
