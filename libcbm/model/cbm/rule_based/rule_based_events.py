import numpy as np
from types import SimpleNamespace


def create_proportional_rule_based_event(rule_filter, proportion):
    pass


def create_sorted_rule_based_event(rule_filter, target):
    rule_based_event = SimpleNamespace()

    return rule_based_event


def apply_rule_based_event(classifiers, inventory, state_variables,
                           undisturbed, rule_based_event,
                           evaluate_filter_func, target_func):

    # returns a boolean numpy array where true indicates the stand is eligible
    # for disturbance
    filtered = evaluate_filter_func(rule_based_event.rule_filter)

    # remove those stands affected by a previous disturbance from eligibility
    filtered = np.logical_and(undisturbed, filtered)

    target = target_func(inventory)
    target_index = target["disturbed_indices"]
    target_area_proportions = target["area_proportions"]

    updated_inventory = inventory.copy()
    updated_classifiers = classifiers.copy()
    updated_state_variables = state_variables.copy()

    split_index = target_index[target_area_proportions < 1.0]
    split_inventory = updated_inventory.iloc[split_index].copy()
    # reduce the area of the disturbed inventory by the disturbance area
    # proportion
    updated_inventory.area = \
        updated_inventory.area * target_area_proportions[split_index]

    # set the split inventory as the remaining undisturbed area
    split_inventory.area = \
        split_inventory.area * (1-target_area_proportions[split_index])

    # create the updated inventory by appending the split records
    updated_inventory = updated_inventory.append(split_inventory)

    # since classifiers and state variables are not altered here (this is done
    # in the model) splitting is just a matter of adding a copy of the split
    # values
    updated_classifiers = updated_classifiers.append(
        updated_classifiers.iloc[split_index].copy())
    updated_state_variables = updated_state_variables.append(
        updated_state_variables[split_index].copy())

    return (target_index, updated_inventory, updated_classifiers,
            updated_state_variables)





