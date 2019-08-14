import pandas as pd


def diff_for_identifier(diffs, identifier):
    """Get differences for all timesteps for a particular identifier.

    Args:
        diffs (pandas.DataFrame): a dataframe containing comparisons.
            See: :py:func:`join_result`
        identifier (int): an integer identifier correspsonding to a
            specific test case

    Returns:
        (pandas.DataFrame): differences for all timesteps
            for a particular identifier.
    """
    diffs = diffs.copy()
    diffs = diffs[diffs["identifier"] == id]
    diffs = diffs.drop(columns="identifier")
    return diffs.groupby("timestep").sum()


def values_for_identifier(values, identifier):
    """Get values for an identifier associated with a particular test case.

    Args:
        values (pandas.DataFrame): [description]
        identifier ([type]): [description]

    Returns:
        [type]: [description]
    """
    values = values.copy()
    values = values[values["identifier"] == id]
    values = values.drop(columns="identifier")
    return values.groupby("timestep").sum()


def summarize_diffs_by_identifier(diffs, result_limit=20):
    """Sort and summarize a diff result returned by :py:func:`join_result`

    Args:
        diffs (pandas.DataFrame): a dataframe containing comparisons
        result_limit (int, optional): The number of rows in the result.
            Defaults to 20.

    Returns:
        pandas.DataFrame: a summary of a difference dataframe sorted
        descending by the "abs_total_diff" column/
    """
    diffs = diffs.copy()
    diffs = diffs.drop(columns="timestep")
    return diffs \
        .groupby("identifier").sum() \
        .sort_values("abs_total_diff", ascending=False) \
        .head(result_limit)


def merge_result(cbm3_result, libcbm_result, value_cols):
    """Produce a merge table for CBM-CFS3 values versus LibCBM values.

    The pair of pandas dataframes have the following columns:

        - "identifier": the identifier for the a set of timeseries values in
            the DataFrame
        - "timestep": the time step for the associated identifier
        - An ordered set of value columns matching the "value_cols" list of
            names.

    Args:
        cbm3_result (pandas.DataFrame): The CBM result to compare
        libcbm_result (pandas.DataFrame): The LibCBM result to compare
        value_cols (list): The list of string names for the values in the pair
            of specified dataframes.

    Returns:
        pandas.DataFrame: dataframe which is the merge of the libcbm value
            with the CBM-CFS3 value by timestep, identifier

    """

    libcbm_result = libcbm_result[['identifier', 'timestep']+value_cols]
    cbm3_result = cbm3_result[['identifier', 'timestep']+value_cols]
    merged = libcbm_result.merge(
        cbm3_result,
        left_on=['identifier', 'timestep'],
        right_on=['identifier', 'timestep'],
        suffixes=("_libCBM", "_cbm3"))

    return merged


def diff_result(merged, value_cols):
    """Produce a diff table for a merged of CBM-CFS3 values versus LibCBM values.

    Args:
        merged (pandas.DataFrame): A merged CBM3/LibCBM comparison as produced
            by: :py:func:`merge_result`.
       value_cols (list): The list of string names for the values in the pair
            of specified dataframes.

    Returns:
        pandas.DataFrame: dataframe which is the comparison of the libcbm
            values with the CBM-CFS3 values by timestep, identifier
    """
    diffs = pd.DataFrame()
    diffs["identifier"] = merged["identifier"]
    diffs["timestep"] = merged["timestep"]
    diffs["abs_total_diff"] = 0
    for flux in value_cols:
        l = "{}_libCBM".format(flux)
        r = "{}_cbm3".format(flux)
        diffs[flux] = (merged[l] - merged[r])
        diffs["abs_total_diff"] += diffs[flux].abs()
    return diffs