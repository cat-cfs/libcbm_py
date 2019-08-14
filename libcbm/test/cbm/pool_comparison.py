import collections
import pandas as pd


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


def prepare_cbm3_pools(cbm3_pools):
    """Readies a cbm3 pools query result for joining with a LibCBM pools
    result.

    Also performs the following table changes to make it easy to join and
    compare with to the libcbm result:

        - rename "TimeStep" to "timestep"
        - convert the "identifier" column to numeric from string
        - rename the columns from cbm3 pool names to libcbm pool names

    Args:
        cbm3_pools (pandas.DataFrame): The CBM-CFS3 pool indicator result

    Returns:
        pandas.DataFrame: a copy of the input dataFrame ready to join with
            libcbm pool results.
    """
    result = cbm3_pools.copy()
    result = result.rename(columns={'TimeStep': 'timestep'})
    result["identifier"] = pd.to_numeric(result["identifier"])
    pool_map = collections.OrderedDict(
        zip(get_cbm3_pools(), get_libcbm_pools()))
    result = result.rename(columns=pool_map)
    return result
