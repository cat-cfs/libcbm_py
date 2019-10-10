import numpy as np


def process_event(filter_factory, classifiers_filter_factory, filter_data,
                  undisturbed, target_func, pools, state_variables,
                  classifiers, inventory):

    filter_result = apply_filter(
        filter_factory, classifiers_filter_factory, filter_data,
        undisturbed, pools, state_variables, classifiers)

    filtered_inventory = inventory[filter_result]
    filtered_state_variables = state_variables[filter_result]
    filtered_pools = pools[filter_result]

    target = apply_rule_based_target(
        target_func, filtered_pools, filtered_inventory,
        filtered_state_variables)

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


def apply_rule_based_target(target_func, pools, inventory, state_variables):
    return target_func(pools, inventory, state_variables)


def apply_rule_based_event(target, classifiers, inventory, pools,
                           state_variables):

    target_index = target["disturbed_indices"]
    target_area_proportions = target["area_proportions"]

    updated_inventory = inventory.copy()
    updated_classifiers = classifiers.copy()
    updated_state_variables = state_variables.copy()
    updated_pools = pools.copy()

    split_index = target_index[target_area_proportions < 1.0]
    split_inventory = updated_inventory.iloc[split_index].copy()
    # reduce the area of the disturbed inventory by the disturbance area
    # proportion
    updated_inventory.area = updated_inventory.area.multiply(
        target_area_proportions[split_index], axis=0)

    # set the split inventory as the remaining undisturbed area
    split_inventory.area = split_inventory.area.multiply(
        1.0 - target_area_proportions[split_index], axis=0)

    # create the updated inventory by appending the split records
    updated_inventory = updated_inventory.append(split_inventory)

    # since classifiers, pools, and state variables are not altered here (this
    # is done in the model) splitting is just a matter of adding a copy of the
    # split values
    updated_classifiers = updated_classifiers.append(
        updated_classifiers.iloc[split_index].copy())
    updated_state_variables = updated_state_variables.append(
        updated_state_variables[split_index].copy())
    updated_pools = updated_pools.append(
        updated_pools[split_index].copy())
    return (target_index, updated_inventory, updated_classifiers,
            updated_state_variables)
