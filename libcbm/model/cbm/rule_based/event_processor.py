import numpy as np


def process_event(filter_factory, classifiers_filter_factory, filter_data,
                  undisturbed, target_func, pools, state_variables,
                  classifiers, inventory):
    """Computes a CBM rule based event by filtering and targeting a subset of
    the specified inventory.  In the case of merchantable or area targets
    splits may occur to meet a disturbance target exactly.

    Args:
        filter_factory (object): collection of methods to create and apply
            filters for CBM stands.
            See :py:mod:`libcbm.model.cbm.rule_based.rule_filter`
        classifiers_filter_factory (object): collection of methods to filter
            stands by classifier values using classifier sets.
            See: :py:class:`libcbm.model.cbm.rule_based.classifier_filter.ClassifierFilter`
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
        pools (pandas.DataFrame): CBM simulation pools
        state_variables (pandas.DataFrame): CBM simulation state variables
        classifiers (pandas.DataFrame): CBM classifier values
        inventory (pandas.DataFrame): CBM inventory

    Returns:
        tuple: The computed disturbance index and pools, state variables,
            classifiers and inventory which can be modified when splitting
            occurs. See the return value of :py:func:`apply_rule_based_event`
    """
    filter_result = apply_filter(
        filter_factory, classifiers_filter_factory, filter_data,
        undisturbed, pools, state_variables, classifiers)

    filtered_inventory = inventory[filter_result]
    filtered_state_variables = state_variables[filter_result]
    filtered_pools = pools[filter_result]

    target = target_func(
        filtered_pools, filtered_inventory, filtered_state_variables)

    return apply_rule_based_event(
        target, classifiers, inventory, pools, state_variables)


def apply_filter(filter_factory, classifiers_filter_factory, filter_data,
                 undisturbed, pools, state_variables, classifiers):

    classifier_filter = classifiers_filter_factory(
        filter_data.classifier_set, classifiers)

    merged_filter = filter_factory.merge_filters(
        filter_factory.create_filter(
            expression=filter_data.pool_filter_expression,
            data=pools,
            columns=filter_data.pool_filter_columns),
        filter_factory.create_filter(
            expression=filter_data.state_filter_expression,
            data=state_variables,
            columns=filter_data.state_filter_columns),
        classifier_filter)
    filter_result = filter_factory.evaluate_filter(merged_filter)

    # set to false those stands affected by a previous disturbance from
    # eligibility
    filter_result = np.logical_and(undisturbed, filter_result)

    return filter_result


def apply_rule_based_event(target, classifiers, inventory, pools,
                           state_variables):

    target_index = target["disturbed_index"]
    target_area_proportions = target["area_proportions"]

    updated_inventory = inventory.copy()
    updated_classifiers = classifiers.copy()
    updated_state_variables = state_variables.copy()
    updated_pools = pools.copy()

    split_index = target_index[target_area_proportions < 1.0]
    split_inventory = updated_inventory.iloc[split_index].copy()
    if len(split_inventory.index) > 0:
        # reduce the area of the disturbed inventory by the disturbance area
        # proportion
        updated_inventory.area = updated_inventory[split_index].area.multiply(
            target_area_proportions, axis=0)

        # set the split inventory as the remaining undisturbed area
        split_inventory.area = split_inventory[split_index].area.multiply(
            1.0 - target_area_proportions, axis=0)

        # create the updated inventory by appending the split records
        updated_inventory = updated_inventory.append(split_inventory)

        # Since classifiers, pools, and state variables are not altered here
        # (this is done in the model) splitting is just a matter of adding a
        # copy of the split values.
        updated_classifiers = updated_classifiers.append(
            updated_classifiers.iloc[split_index].copy())
        updated_state_variables = updated_state_variables.append(
            updated_state_variables[split_index].copy())
        updated_pools = updated_pools.append(
            updated_pools[split_index].copy())
    return (updated_classifiers, updated_inventory, pools,
            updated_state_variables)
