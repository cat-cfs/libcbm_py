# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pandas as pd


def get_libcbm_result_suffix():
    """Returns a suffix for use in the column names of cbm3/libcbm comparison
    dataframes

    Returns:
        str: "_libcbm"
    """
    return "_libcbm"


def get_cbm3_result_suffix():
    """Returns a suffix for use in the column names of cbm3/libcbm comparison
    dataframes

    Returns:
        str: "_cbm3"
    """
    return "_cbm3"


def summarize_diffs_by_identifier(diffs, result_limit=None):
    """Sort and summarize a diff result returned by :py:func:`join_result`

    Args:
        diffs (pandas.DataFrame): a dataframe containing comparisons
        result_limit (int, optional): The number of rows in the result.
            If set to None all rows are returned. Defaults to None.

    Returns:
        pandas.DataFrame: a summary of a difference dataframe sorted
        descending by the "abs_total_diff" column/
    """
    diffs = diffs.copy()
    diffs = diffs.drop(columns="timestep")
    result = diffs \
        .groupby("identifier").sum() \
        .sort_values("abs_total_diff", ascending=False)
    if result_limit:
        result = result.head(result_limit)
    return result


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
        suffixes=(get_libcbm_result_suffix(), get_cbm3_result_suffix()))

    return merged


def diff_result(merged):
    """Produce a diff table for a merge of CBM-CFS3 values versus LibCBM
    values.

    Args:
        merged (pandas.DataFrame): A merged CBM3/LibCBM comparison as produced
            by: :py:func:`merge_result`.

    Returns:
        pandas.DataFrame: dataframe which is the comparison of the libcbm
            values with the CBM-CFS3 values by timestep, identifier
    """
    diffs = pd.DataFrame()
    if "identifier" in merged.columns:
        diffs["identifier"] = merged["identifier"]
    diffs["timestep"] = merged["timestep"]
    diffs["abs_total_diff"] = 0
    all_col = list(merged.columns)
    value_cols = [x for x in all_col if x.endswith("_cbm3")]
    value_cols = [x[:-(len("_cbm3"))] for x in value_cols]
    for flux in value_cols:
        left = f"{flux}{get_libcbm_result_suffix()}"
        right = f"{flux}{get_cbm3_result_suffix()}"
        diffs[flux] = (merged[left] - merged[right])
        diffs["abs_total_diff"] += diffs[flux].abs()
    return diffs


def get_summarized_diff_plot(merged, max_results, x_label, y_label,
                             **plot_kwargs):
    """Produce a plot of libcbm vs cbm3 merged results including all test
    cases.  Passes any additional key word args to the pandas.DataFrame.plot
    function.


    Args:
        merged (pandas.DataFrame): A merged CBM3/LibCBM comparison as produced
            by: :py:func:`merge_result`.
        max_results (int): The maximum number of difference summaries to plot.
            None for unlimited.
        x_label (str): The label on the x axis of the resulting plot
        y_label (str): The label on the y axis of the resulting plot

    Returns:
        matplotlib.AxesSubplot or np.array: the return value of
            pandas.DataFrame.plot
    """
    diff = diff_result(merged)
    summarized_diff = summarize_diffs_by_identifier(
        diff, max_results)
    ax = summarized_diff.plot(
        **plot_kwargs)
    ax.set(
        xlabel=x_label,
        ylabel=y_label)
    return ax


def get_test_case_comparison_plot(identifier, merged, diff, x_label, y_label,
                                  **plot_kwargs):
    """Gets a comparison plot for a CBM3 simulation versus a LibCBM simulation
    for a single test case, or the summary for all test cases in merged if the
    identifier is not specified

    Args:
        identifier (int): The test case id, if set to none, the comparison
            includes a summarized view of all identifiers in the specified
            merged DataFrame.
        merged (pandas.DataFrame): A merged CBM3/LibCBM comparison as produced
            by: :py:func:`merge_result`.
        diff (bool): if true return differences from the merge, and otherwise
            return the raw merged values
        x_label (str): The label on the x axis of the resulting plot
        y_label (str): The label on the y axis of the resulting plot

    Returns:
        matplotlib.AxesSubplot or np.array: the return value of
            pandas.DataFrame.plot
    """
    markers = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P",
               "*", "h", "H", "+", "x", "X", "D", "d"]
    if identifier:
        subset = merged[merged["identifier"] == identifier].copy()
    else:
        subset = merged.copy()
    subset = subset.drop(columns="identifier")

    if diff:
        subset = diff_result(subset)
    subset = subset.groupby("timestep").sum()
    ax = subset.plot(
        **plot_kwargs)
    ax.set(
        xlabel=x_label,
        ylabel=y_label)
    for i, line in enumerate(ax.get_lines()):
        line.set_marker(markers[i % len(markers)])
    ax.legend(ax.get_lines(), subset.columns, loc='best')
    return ax
