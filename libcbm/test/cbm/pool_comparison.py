import collections
import pandas as pd
from libcbm.test.cbm import result_comparison


def get_libcbm_pools():
    """Gets the names of all libcbm pools as a list

    Returns:
        list: all libcbm pool names
    """
    return get_libcbm_biomass_pools() + get_libcbm_dom_pools()


def get_cbm3_pools():
    """Gets the names of all CBM-CFS3 pools as a list

    Returns:
        list: all CBM-CFS3 pool names
    """
    return get_cbm3_biomass_pools() + get_cbm3_dom_pools()


def get_cbm3_biomass_pools():
    """Gets the names of all CBM-CFS3 biomass pools as a list

    Returns:
        list: all libcbm biomass pool names
    """
    return [
        'Softwood Merchantable', 'Softwood Foliage', 'Softwood Other',
        'Softwood Coarse Roots', 'Softwood Fine Roots',
        'Hardwood Merchantable', 'Hardwood Foliage', 'Hardwood Other',
        'Hardwood Coarse Roots', 'Hardwood Fine Roots']


def get_cbm3_dom_pools():
    """Gets the names of all CBM-CFS3 dom pools as a list

    Returns:
        list: all libcbm CBM-CFS3 dom pool names
    """
    return [
        'Aboveground Very Fast DOM', 'Belowground Very Fast DOM',
        'Aboveground Fast DOM', 'Belowground Fast DOM', 'Medium DOM',
        'Aboveground Slow DOM', 'Belowground Slow DOM', 'Softwood Stem Snag',
        'Softwood Branch Snag', 'Hardwood Stem Snag', 'Hardwood Branch Snag']


def get_libcbm_biomass_pools():
    """Gets the names of all libcbm biomass pools as a list

    Returns:
        list: all libcbm libcbm biomass pool names
    """
    return [
        'SoftwoodMerch', 'SoftwoodFoliage', 'SoftwoodOther',
        'SoftwoodCoarseRoots', 'SoftwoodFineRoots', 'HardwoodMerch',
        'HardwoodFoliage', 'HardwoodOther', 'HardwoodCoarseRoots',
        'HardwoodFineRoots']


def get_libcbm_dom_pools():
    """Gets the names of all libcbm dom pools as a list

    Returns:
        list: all libcbm libcbm dom pool names
    """
    return [
        'AboveGroundVeryFastSoil', 'BelowGroundVeryFastSoil',
        'AboveGroundFastSoil', 'BelowGroundFastSoil', 'MediumSoil',
        'AboveGroundSlowSoil', 'BelowGroundSlowSoil', 'SoftwoodStemSnag',
        'SoftwoodBranchSnag', 'HardwoodStemSnag', 'HardwoodBranchSnag']


def get_pool_map(name):
    """Returns a mapping between libcbm poolnames and cbm-cfs3 poolnames.
    Assumes the pools are ordered.
    Arguments:
        name (str): one of:
            - "all" for a map of all pools
            - "biomass" for a map of all biomass pools
            - "dom" for a map of all dead organic matter pools
    Raises:
        ValueError: the name parameter was not supported
    Returns:
        collections.OrderedDict: mapping of CBM-CFS3 pools to libcbm pools
    """
    if name == "all":
        return collections.OrderedDict(
            zip(get_cbm3_pools(), get_libcbm_pools()))
    elif name == "biomass":
        return collections.OrderedDict(
            zip(get_cbm3_biomass_pools(), get_libcbm_biomass_pools()))
    elif name == "dom":
        return collections.OrderedDict(
            zip(get_cbm3_dom_pools(), get_libcbm_dom_pools()))
    else:
        raise ValueError(
                "unknown comparison name, expected 'all', 'biomass', or 'dom'")


def prepare_cbm3_pools(cbm3_pools, pool_map):
    """Readies a cbm3 pools query result for joining with a LibCBM pools
    result.

    Also performs the following table changes to make it easy to join and
    compare with to the libcbm result:

        - rename "TimeStep" to "timestep"
        - convert the "identifier" column to numeric from string
        - rename the columns from cbm3 pool names to libcbm pool names

    Args:
        cbm3_pools (pandas.DataFrame): The CBM-CFS3 pool indicator result
        pool_map: (collections.OrderedDict): a map of pools

    Returns:
        pandas.DataFrame: a copy of the input dataFrame ready to join with
            libcbm pool results.
    """
    result = cbm3_pools.copy()
    result = result.rename(columns={'TimeStep': 'timestep'})
    result["identifier"] = pd.to_numeric(result["identifier"])
    result = result.rename(columns=pool_map)
    return result


def get_merged_pools(cbm3_pools, libcbm_pools, pools_included="all"):
    """Produces a merge of the specified cbm3 and libcbm pool results

    Args:
        cbm3_pools (pandas.DataFrame): cbm3 pool results as produced by:
            :py:func:`libcbm.test.cbm.cbm3_support.cbm3_simulator.get_cbm3_results`
        libcbm_pools (pandas.DataFrame): libcbm pool results as produced by:
            :py:func:`libcbm.test.cbm.test_case_simulator.run_test_cases`
        name (str): one of:
            - "all" for a map of all pools
            - "biomass" for a map of all biomass pools
            - "dom" for a map of all dead organic matter pools
        pools_included (str, optional): one of:

                - "all" to merge all pools
                - "biomass" to merge all biomass pools
                - "dom" to merge all dead organic matter pools.

            Defaults to "all".

    Returns:
        pandas.DataFrame: merged comparison of CBM3 versus libcbm for analysis
    """
    pool_map = get_pool_map(pools_included)
    merged_pools = result_comparison.merge_result(
        prepare_cbm3_pools(cbm3_pools, pool_map),
        libcbm_pools,
        list(pool_map.values()))
    return merged_pools
