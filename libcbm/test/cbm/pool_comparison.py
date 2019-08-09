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


def get_pool_map(name):
    """Returns a mapping between libcbm poolnames and cbm-cfs3 poolnames.
    Assumes the pools are ordered.

    Arguments:
        name (str): one of "all" for a map of all pools, or "biomass"
            for a map of all biomass pools

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
    else:
        raise ValueError(
                "unknown comparison name, expected 'all' or 'biomass'")


def join_pools(libCBM_pools, cbm3_pools, comparison):
    """Produce 2 dataframes used to compare CBM-CFS3 pool results versus
    LibCBM pool results.

    Arguments:
        libCBM_pools (pandas.DataFrame): dataframe whose values are libcbm
            simulation results. Contains columns:

                - "identifier"
                - "timestep"
                - all libcbm pool names
        cbm3_pools (pandas.DataFrame): dataframe whose values are CBM-CFS3
            simulation results. Contains columns:

                - "identifier"
                - "Timestep"
                - all cbm-cfs3 pool names
        comparison (str): a string value that is one of:

            - "all" to include all pools in the resulting comparison
            - "biomass" to include only biomass pools in the resulting
              comparison

    Returns:
        Tuple: A tuple containing comparison data:

            - value1: dataframe which is the merge of the libcbm pools with
                    the CBM-CFS3 pools
            - value2: dataframe which is the comparison of the libcbm pools
                with the CBM-CFS3 pools
    """
    pool_mapping = get_pool_map(comparison)
    libCBM_poolnames = list(pool_mapping.values())
    cbm3_poolnames = list(pool_mapping.keys())
    libCBM_pools = libCBM_pools[['identifier', 'timestep']+libCBM_poolnames]
    cbm3_pools = cbm3_pools.rename(columns={'TimeStep': 'timestep'})
    cbm3_pools = cbm3_pools[['identifier', 'timestep']+cbm3_poolnames]

    # make column naming consistent
    cbm3_pools = cbm3_pools.rename(columns=pool_mapping)

    merged = libCBM_pools.merge(
            cbm3_pools,
            left_on=['identifier', 'timestep'],
            right_on=['identifier', 'timestep'],
            suffixes=("_libCBM", "_cbm3"))

    # compute diffs row-by-row
    diffs = pd.DataFrame()
    diffs["identifier"] = merged["identifier"]
    diffs["timestep"] = merged["timestep"]
    diffs["abs_total_diff"] = 0
    for pool in libCBM_poolnames:
        l = "{}_libCBM".format(pool)
        r = "{}_cbm3".format(pool)
        diffs[pool] = (merged[l] - merged[r])
        diffs["abs_total_diff"] += diffs[pool].abs()

    return merged, diffs
