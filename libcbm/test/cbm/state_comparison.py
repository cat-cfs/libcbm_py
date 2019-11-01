"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import pandas as pd
from libcbm.test.cbm import result_comparison


def prepare_cbm3_state(cbm3_state):
    """Readies a cbm3 state query result for joining with a LibCBM state
    result.  Since CBM3 does not have external access to much of its
    internal simulation state this is currently limited to only comparing age,
    and land_class

    Also performs the following table changes to make it easy to join and
    compare with to the libcbm result:

        - rename:

            - "TimeStep" to "timestep"
            - "LandClassID" to "land_class"
            - "Average Age" to "age"

        - convert the "identifier" column to numeric from string

    Args:
        cbm3_pools (pandas.DataFrame): The CBM-CFS3 pool indicator result
        pool_map: (collections.OrderedDict): a map of pools

    Returns:
        pandas.DataFrame: a copy of the input dataFrame ready to join with
            libcbm pool results.
    """
    result = cbm3_state.copy()
    result = result.rename(columns={
        'TimeStep': 'timestep',
        'LandClassID': 'land_class',
        'Average Age': "age"
        # average age to age is OK math since we are dealing with single stand
        # simulations and not strata
        })
    result["identifier"] = pd.to_numeric(result["identifier"])
    result = result[["identifier", "timestep", "age", "land_class"]]
    return result


def get_merged_state(cbm3_state, libcbm_state):
    """merges CBM3 and libcbm state variables for comparison

    Args:
        cbm3_state (pandas.DataFrame): cbm3 state results as produced by:
            :py:func:`libcbm.test.cbm.cbm3_support.cbm3_simulator.get_cbm3_results`
        libcbm_state (pandas.DataFrame): libcbm state results as produced by:
            :py:func:`libcbm.test.cbm.test_case_simulator.run_test_cases`

    Returns:
        pandas.DataFrame: merged comparison of CBM3 versus libcbm for analysis
    """
    merged_state = result_comparison.merge_result(
        prepare_cbm3_state(cbm3_state),
        libcbm_state,
        ["age", "land_class"])
    return merged_state
