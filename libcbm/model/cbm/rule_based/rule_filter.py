# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


from types import SimpleNamespace
import numexpr
import numpy as np


def create_filter(expression, data):
    """Creates a filter object for filtering a pandas dataframe using an
    expression.

    Args:
        expression (str): a boolean expression in terms of the values in
            column_variable_map
        data (pandas.DataFrame, dict or object): the data to for which to
            create a filter

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
    result.local_dict = data
    return result


def evaluate_filters(*filter_objs):
    """evaluates the specified sequence of filter object

    Args:
        filter_obj (list): list of objects with properties:

            - expression (str): a boolean expression to filter the values
                in local_dict. The variables are defined as the keys in
                local_dict.
            - local_dict (dict): a dictionary containing named numpy
                variables to filter.

    Returns:
        np.ndarray: filter result

    """
    output = True
    for filter_obj in filter_objs:
        if not filter_obj or not filter_obj.expression:
            continue
        result = numexpr.evaluate(filter_obj.expression, filter_obj.local_dict)
        if output is True:
            output = result
        else:
            output = np.logical_and(output, result)

    return output
