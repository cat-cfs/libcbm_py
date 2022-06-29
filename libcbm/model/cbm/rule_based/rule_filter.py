# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import Union
from libcbm.storage import dataframe
from libcbm.storage.series import Series
from libcbm.storage.dataframe import DataFrame


class RuleFilter:
    def __init__(self, expression: str, data: DataFrame):
        self._expression = expression
        self._data = data

    @property
    def expression(self) -> str:
        """
        a boolean expression to filter the values
        in self.data.
        """
        return self._expression

    @property
    def data(self) -> DataFrame:
        """
        a table containing columns to filter.
        """
        return self._data


def create_filter(expression: str, data: DataFrame):
    """Creates a filter object for filtering a pandas dataframe using an
    expression.

    Args:
        expression (str): a boolean expression in terms of the values in
            column_variable_map
        data: the data to for which to create a filter

    Returns:
        RuleFilter: rule_filter object
    """

    return RuleFilter(expression, data)


def evaluate_filters(*filter_objs: RuleFilter) -> Union[Series, None]:
    """Evaluates the specified sequence of filter objects.

    * If all filter expressions in the specified filter_objs are null then a
      True (unfiltered) series is returned
    * If all filter expressions in the specified filter_objs are null and no
      data is provided None is returned
    * otherwise the series logical and of all specified filters are returned.

    Args:
        filter_objs (list): list of RuleFilter objects:

    Returns:
        Series: filter result (boolean array)
    """
    out_series_length = None
    out_series_backend_type = None
    output = None

    for filter_obj in filter_objs:
        if filter_obj and filter_obj.data:
            if not out_series_length:
                out_series_length = filter_obj.data.n_rows
                out_series_backend_type = filter_obj.data.backend_type
            elif out_series_length != filter_obj.data.n_rows:
                raise ValueError("data length mismatch")

        if not filter_obj or not filter_obj.expression or not filter_obj.data:
            continue

        result = filter_obj.data.evaluate_filter(filter_obj.expression)

        if output is None:
            output = result
        else:
            output = dataframe.logical_and(output, result)

    if not output and out_series_length:
        output = dataframe.make_boolean_series(
            True, out_series_length, out_series_backend_type
        )
    return output
