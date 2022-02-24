# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


from types import SimpleNamespace
import pandas as pd
import numpy as np
import ctypes


# def _make_contiguous(df):
#     """Orders the underlying memory in a numpy-backed dataframe as C contiguous
#     (row major ordering)
#
#     Args:
#         df (DataFrame): a dataframe
#
#     Returns:
#         DataFrame: a C contiguous copy of the input data frame.
#     """
#
#     if not df.values.flags["C_CONTIGUOUS"]:
#         return pd.DataFrame(
#             columns=df.columns.tolist(), data=np.ascontiguousarray(df)
#         )
#     return df
#
#
# def prepare(cbm_vars: CBMVariables) -> CBMVariables:
#     """prepares, validates the specified cbm_vars object for use with low
#     level functions
#
#     Args:
#         cbm_vars (CBMVariables): the cbm variables to validate and prepare
#     """
#
#     for field in ["pools", "flux", "classifiers"]:
#         if field in cbm_vars.__dict__:
#             cbm_vars.__dict__[field] = _make_contiguous(
#                 cbm_vars.__dict__[field]
#             )
#
#     return cbm_vars


def unpack_ndarrays(variables):
    """Convert and return a set of variables as a types.SimpleNamespace whose
    members are only ndarray.
    Supports 2 cases:

        1: the specified variables are already stored in a SimpleNamespace
           whose properties are one of pd.Series,pd.DataFrame,or np.ndarray.
           For each property in the namespace, convert to an ndarray if
           necessary.
        2: the specified variables are the columns of a pandas.DataFrame.
           Return a reference to each column's underlying numpy.ndarray storage

    Args:
        variables (SimpleNamespace, pd.DataFrame): The set of variables to
        unpack.

    Raises:
        ValueError: the type of the specified argument was not supported

    Returns:
        types.SimpleNamespace: a SimpleNamespace whose properties are ndarray.
    """
    properties = {}
    if isinstance(variables, SimpleNamespace):
        for k, v in variables.__dict__.items():
            properties[k] = get_ndarray(v)
    elif isinstance(variables, pd.DataFrame):
        for c in variables:
            properties[c] = get_ndarray(variables[c])
    else:
        raise ValueError("Unsupported type")
    return SimpleNamespace(**properties)


def get_ndarray(a):
    """Helper method to deal with numpy arrays stored in pandas objects.
    Gets a reference to the underlying numpy.ndarray storage from a
    pandas.DataFrame or pandas.Series.

    Args:
        a (None, ndarray, pandas.DataFrame, or pandas.Series): data to
            potentially convert to ndarray

    Returns:
        any: the specified value, or the ndarray storage of a
            specified pandas object
    """
    if isinstance(a, pd.DataFrame) or isinstance(a, pd.Series):
        return a.values
    else:
        return a
