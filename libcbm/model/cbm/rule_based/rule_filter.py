from types import SimpleNamespace
import numexpr


def intersection(first, *others):
    """Intersect all parameters
    https://stackoverflow.com/questions/2953280/taking-intersection-of-n-many-lists-in-python

    Args:
        first (iterable): the seed iterable

    Returns:
        set: the intersection of all parameters
    """
    return set(first).intersection(*others)


def merge_filters(filters):
    """Merge filters into a single filter

    Args:
        filters (list): a list of objects with properties:
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
    intersecting_variables = intersection(
        filters[0].local_dict.keys(),
        *[f.local_dict.keys() for f in filters[1:]])
    if intersecting_variables:
        raise ValueError(
            "The following variables are present in more than one filter "
            f"{intersecting_variables}")
    result = SimpleNamespace()
    result.expression = "({})".format(
        ") & (".join([x.expression for x in filters if x.expression]))
    result.local_dict = {}
    for f in filters:
        if f.expression:
            result.local_dict.update(f.local_dict)


def create_filter(expression, data, columns, column_variable_map=None):
    """Creates a filter object for filtering a pandas dataframe using an
    expression.

    Args:
        expression (str): a boolean expression in terms of the values in
            column_variable_map
        data (pandas.DataFrame): the data to for which to create a filter
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
    if filter_obj.expression:
        return numexpr.evaluate(filter_obj.expression, filter_obj.local_dict)
    return None
