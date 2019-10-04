from libcbm.model.cbm.rule_based import rule_filter
from libcbm.model.cbm.rule_based.classifier_filter import ClassifierFilter

# TEMPORARY: see issue #15 (https://github.com/cat-cfs/libcbm_py/issues/15)
from libcbm.test.cbm import pool_comparison
#########


def get_pool_variable_filter_mappings():

    biomass = pool_comparison.get_libcbm_biomass_pools()
    return [
        ("MinTotBiomassC", biomass, ">="),
        ("MaxTotBiomassC", biomass, "<="),
        ("MinSWMerchBiomassC", ["SoftwoodMerch"], ">="),
        ("MaxSWMerchBiomassC", ["SoftwoodMerch"], "<="),
        ("MinHWMerchBiomassC", ["HardwoodMerch"], ">="),
        ("MaxHWMerchBiomassC", ["HardwoodMerch"], "<="),
        ("MinTotalStemSnagC", ["SoftwoodStemSnag", "HardwoodStemSnag"], ">="),
        ("MaxTotalStemSnagC", ["SoftwoodStemSnag", "HardwoodStemSnag"], "<="),
        ("MinSWStemSnagC", ["SoftwoodStemSnag"], ">="),
        ("MaxSWStemSnagC", ["SoftwoodStemSnag"], "<="),
        ("MinHWStemSnagC", ["HardwoodStemSnag"], ">="),
        ("MaxHWStemSnagC", ["HardwoodStemSnag"], "<="),
        ("MinTotalStemSnagMerchC",
         ["SoftwoodMerch", "HardwoodMerch",
          "SoftwoodStemSnag", "HardwoodStemSnag"], ">="),
        ("MaxTotalStemSnagMerchC",
         ["SoftwoodMerch", "HardwoodMerch",
          "SoftwoodStemSnag", "HardwoodStemSnag"], "<="),
        ("MinSWMerchStemSnagC", ["SoftwoodMerch", "SoftwoodStemSnag"], ">="),
        ("MaxSWMerchStemSnagC", ["SoftwoodMerch", "SoftwoodStemSnag"], "<="),
        ("MinHWMerchStemSnagC", ["HardwoodMerch", "HardwoodStemSnag"], ">="),
        ("MaxHWMerchStemSnagC", ["HardwoodMerch", "HardwoodStemSnag"], "<=")]


def create_pool_value_filter(sit_data):
    """Create a filter against simulation pool values based on a single
    row of SIT disturbance event, or transition rule data.

    Args:
        sit_data (dict): a row dictionary from an SIT events, or SIT
            transition rules table

    Returns:
        tuple:
            1. expression (str): a boolean expression compatible with numexpr
            2. columns (list): the list of column names that are referenced in
                the expression

            These values are intended for use with the
            :py:func:`libcbm.model.rule_based.rule_filter module`
    """
    columns = set()
    expression_tokens = []

    mappings = get_pool_variable_filter_mappings()
    for sit_column, pool_values, operator in mappings:
        if sit_data[sit_column] < 0:
            # by convention, SIT criteria less than 0 are considered null
            # criteria
            continue
        columns.update(set(pool_values))
        expression_tokens.append(
            "({pool_expression} {operator} {value})".format(
                pool_expression="({})".format(" + ".join(pool_values)),
                operator=operator,
                value=sit_data[sit_column]
            ))

    expression = " & ".join(expression_tokens)
    return expression, columns


def get_state_variable_age_filter_mappings():
    """get mappings between SIT events or transitions age criteria columns,
    and state variable columns, along with a boolean operator to compare age
    values.

    Returns:
        list: a list of (str, str, str) tuples in format

             - SIT_Event column name
             - state variable column name
             - operator string
    """
    return [
        ("min_age", "age", ">="),
        ("max_age", "age", "<=")]


def get_state_variable_filter_mappings():
    """get mappings between SIT events criteria columns, and state variable
    columns, along with a boolean operator to compare values.

    Returns:
        list: a list of (str, str, str) tuples in format

             - SIT_Event column name
             - state variable column name
             - operator string

    """
    return get_state_variable_age_filter_mappings() + [
        ("MinYearsSinceDist", "time_since_last_disturbance", ">="),
        ("MaxYearsSinceDist", "time_since_last_disturbance", "<="),
        ("LastDistTypeID", "last_disturbance_type", "==")]


def create_state_variable_filter(sit_data, filter_mappings):
    """Create a filter against simulation state variables based on a single
    row of SIT disturbance event, or transition rule data.

    Args:
        sit_data (dict): a row dictionary from an SIT events, or SIT
            transition rules table
        filter_mappings (list): the return value of either:

            - :py:func:`get_state_variable_filter_mappings`
              (for use with sit events)
            - :py:func:`get_state_variable_age_filter_mappings`
              (for use with sit transition rules)

    Returns:
        tuple:
            1. expression (str): a boolean expression compatible with numexpr
            2. columns (list): the list of column names that are referenced in
                the expression

            These values are intended for use with the
            :py:func:`libcbm.model.rule_based.rule_filter module`
    """

    columns = set()
    expression_tokens = []
    for sit_column, state_variable_column, operator in filter_mappings:

        if sit_data[sit_column] < 0:
            # by convention, SIT criteria less than 0 are considered null
            # criteria
            continue
        columns.add(state_variable_column)
        expression_tokens.append(
            "({state_variable} {operator} {value})".format(
                state_variable=state_variable_column,
                operator=operator,
                value=sit_data[sit_column]
            ))

    expression = " & ".join(expression_tokens)
    return expression, columns


def create_classifier_filter(sit_data, classifier_values,
                             classifier_filter_builder):
    classifier_set = [
        sit_data[x] for x in classifier_values.columns.values.tolist()]
    return classifier_filter_builder.create_classifiers_filter(
        classifier_set, classifier_values)
