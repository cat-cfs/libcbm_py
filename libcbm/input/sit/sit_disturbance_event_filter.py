from libcbm.model.cbm.rule_based import rule_filter
from libcbm.model.cbm.rule_based.classifier_filter import ClassifierFilter


def get_pool_variable_filter_mappings():
    return [
        ("MinTotBiomassC", "", ""),
        ("MaxTotBiomassC", "", ""),
        ("MinSWMerchBiomassC", "", ""),
        ("MaxSWMerchBiomassC", "", ""),
        ("MinHWMerchBiomassC", "", ""),
        ("MaxHWMerchBiomassC", "", ""),
        ("MinTotalStemSnagC", "", ""),
        ("MaxTotalStemSnagC", "", ""),
        ("MinSWStemSnagC", "", ""),
        ("MaxSWStemSnagC", "", ""),
        ("MinHWStemSnagC", "", ""),
        ("MaxHWStemSnagC", "", ""),
        ("MinTotalStemSnagMerchC", "", ""),
        ("MaxTotalStemSnagMerchC", "", ""),
        ("MinSWMerchStemSnagC", "", ""),
        ("MaxSWMerchStemSnagC", "", ""),
        ("MinHWMerchStemSnagC", "", ""),
        ("MaxHWMerchStemSnagC", "", "")
   ]


def get_state_variable_filter_mappings():
    """get mappings between SIT events criteria columns, and state variable
    columns, along with a boolean operator to compare values.

    Returns:
        list: a list of (str, str, str) tuples in format

             - SIT_Event column name
             - state variable column name
             - operator string

    """
    return [
        ("min_age", "age", ">="),
        ("max_age", "age", "<="),
        ("MinYearsSinceDist", "time_since_last_disturbance", ">=")
        ("MaxYearsSinceDist", "time_since_last_disturbance", "<=")
        ("LastDistTypeID", "last_disturbance_type", "==")]


def create_state_variable_filter(sit_event, state_variables):
    expression_tokens = []

    expression_tokens = [
        "({state_variable} {operator} {value})".format(
            state_variable=state_variable_column,
            operator=operator,
            value=sit_event[sit_column]
        )
        for sit_column, state_variable_column, operator in
        get_state_variable_filter_mappings()
        if sit_event[sit_column] >= 0]

    expression = " & ".join(expression_tokens)
    return rule_filter.create_filter(
        expression, state_variables, columns=["age"])


def create_classifier_filter(sit_event, classifier_values,
                             classifier_filter_builder):
    classifier_set = [
        sit_event[x] for x in classifier_values.columns.values.tolist()]
    return classifier_filter_builder.create_classifiers_filter(
        classifier_set, classifier_values)

