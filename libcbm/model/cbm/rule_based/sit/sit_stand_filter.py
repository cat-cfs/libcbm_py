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


def create_pool_filter_expression(sit_data):
    """Create a filter against simulation pool values based on a single
    row of SIT disturbance events

    Args:
        sit_data (dict): a row dictionary from an SIT events

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


def create_state_filter_expression(sit_data, age_only):
    """Create a filter against simulation state variables based on a single
    row of SIT disturbance event, or transition rule data.

    Args:
        sit_data (dict): a row dictionary from an SIT events, or SIT
            transition rules table
        age_only (bool, optional): if true specifies that only age variables
            will be considered when building the expression. This is useful
            for SIT_Transition_Rules which consider age as a criteria.
            If False all state variables are included, this is required for
            SIT_Events.

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
    if age_only:
        filter_mappings = get_state_variable_age_filter_mappings()
    else:
        filter_mappings = get_state_variable_filter_mappings()
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


def get_classifier_set(sit_data_row, classifiers):
    """Get a classifier set from a row of SIT data
        disturbance events, transition rules, yield or inventory

    Args:
        sit_data_row (dict): a row dictionary from SIT data
        classifiers (list): list of classifier names

    Returns:
        list: a list of strings with classifier values in the specified row
            also known as a "classifier set"
    """
    classifier_set = [
        sit_data_row[x] for x in classifiers]
    return classifier_set

