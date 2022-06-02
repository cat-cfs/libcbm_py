# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import numexpr
import numpy as np
from typing import Union
from libcbm.storage.series import Series
from libcbm.storage.dataframe import DataFrame


class RuleFilter:
    def __init__(
        self, expression: str, data: DataFrame
    ):
        self._expression = expression
        self._data = data

    @property
    def expression(self) -> str:
        """
        a boolean expression to filter the values
        in local_dict. The variables are defined as the keys in
        data.
        """
        return self._expression

    @property
    def data(self) -> Union[DataFrame, dict[str, Series]]:
        """
        a dictionary or table containing named variables to filter.
        """
        return self._local_dict


def create_filter(expression: str, data: DataFrame):
    """Creates a filter object for filtering a pandas dataframe using an
    expression.

    Args:
        expression (str): a boolean expression in terms of the values in
            column_variable_map
        data: the data to for which to
            create a filter

    Returns:
        RuleFilter: rule_filter object
    """

    return RuleFilter(expression, data)


def evaluate_filters(*filter_objs: RuleFilter) -> Series:
    """evaluates the specified sequence of filter objects

    Args:
        filter_objs (list): list of RuleFilter objects:

    Returns:
        Series: filter result (boolean array)
    """
    output = True
    for filter_obj in filter_objs:
        if not filter_obj or not filter_obj.expression:
            continue
        result = numexpr.evaluate(filter_obj.expression, filter_obj.data)
        if output is True:
            output = result
        else:
            output = np.logical_and(output, result)

    return output
