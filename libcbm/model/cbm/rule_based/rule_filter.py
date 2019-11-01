"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from types import SimpleNamespace
import numexpr


def merge_filters(*filters):
    """Merge filters into a single filter

    Args:
        filters (iterable): a list of objects with properties:

            - expression (str): a boolean expression to filter the values
                in local_dict. The variables are defined as the keys in
                local_dict.
            - local_dict (dict): a dictionary containing named numpy
                variables to filter.

    Raises:
        ValueError: At least 2 of the specified filters contains the same
            variable

    Returns:
        object: the merge of the specified filters, with the same properties
            as the input objects

    """
    if not filters:
        return None

    result = SimpleNamespace()
    result.expression = "({})".format(
        ") & (".join([x.expression for x in filters if x.expression]))
    result.local_dict = {}

    intersecting_variables = set()
    for merge_filter in filters:
        if merge_filter.expression:
            intersecting_variables.update(
                set(result.local_dict.keys()).intersection(
                    set(merge_filter.local_dict.keys())))
            result.local_dict.update(merge_filter.local_dict)

    if intersecting_variables:
        raise ValueError(
            "The following variables are present in more than one filter "
            f"{intersecting_variables}")
    return result


def create_filter(expression, data, columns, column_variable_map=None):
    """Creates a filter object for filtering a pandas dataframe using an
    expression.

    Args:
        expression (str): a boolean expression in terms of the values in
            column_variable_map
        data (pandas.DataFrame, dict or object): the data to for which to
            create a filter
        columns (list): the columns involved in the expression
        column_variable_map (dict, optional): a mapping of the column names
            to variable names in the expression. Default None. If None, the
            column names themselves are assumed to be present in the
            expression.

    Returns:
        object: object with properties:
            - expression (str): a boolean expression to filter the values
                in local_dict. The variables are defined as the keys in
                local_dict.
            - local_dict (dict): a dictionary containing named numpy
                variables to filter.

    """
    result = SimpleNamespace()
    result.expression = expression
    if column_variable_map:
        result.local_dict = {
            column_variable_map[col]: data[col].to_numpy()
            for i, col in enumerate(columns)}
    else:
        result.local_dict = {
            col: data[col].to_numpy()
            for i, col in enumerate(columns)}
    return result


def evaluate_filter(filter_obj):
    """evaluates the specified filter object

    Args:
        filter_obj (object): object with properties:

            - expression (str): a boolean expression to filter the values
                in local_dict. The variables are defined as the keys in
                local_dict.
            - local_dict (dict): a dictionary containing named numpy
                variables to filter.

    Returns:
        np.ndarray: filter result

    """
    if not filter_obj or not filter_obj.expression:
        return None
    return numexpr.evaluate(filter_obj.expression, filter_obj.local_dict)
