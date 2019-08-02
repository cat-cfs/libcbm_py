import collections
import pandas as pd


def get_libcbm_pools():
    return get_libcbm_biomass_pools() + get_libcbm_dom_pools()


def get_cbm3_pools():
    return get_cbm3_biomass_pools() + get_cbm3_dom_pools()


def get_cbm3_biomass_pools():
    return [
        'Softwood Merchantable', 'Softwood Foliage', 'Softwood Other',
        'Softwood Coarse Roots', 'Softwood Fine Roots',
        'Hardwood Merchantable', 'Hardwood Foliage', 'Hardwood Other',
        'Hardwood Coarse Roots', 'Hardwood Fine Roots']


def get_cbm3_dom_pools():
    return [
        'Aboveground Very Fast DOM', 'Belowground Very Fast DOM',
        'Aboveground Fast DOM', 'Belowground Fast DOM', 'Medium DOM',
        'Aboveground Slow DOM', 'Belowground Slow DOM', 'Softwood Stem Snag',
        'Softwood Branch Snag', 'Hardwood Stem Snag', 'Hardwood Branch Snag']


def get_libcbm_biomass_pools():
    return [
        'SoftwoodMerch', 'SoftwoodFoliage', 'SoftwoodOther',
        'SoftwoodCoarseRoots', 'SoftwoodFineRoots', 'HardwoodMerch',
        'HardwoodFoliage', 'HardwoodOther', 'HardwoodCoarseRoots',
        'HardwoodFineRoots']


def get_libcbm_dom_pools():
    return [
        'AboveGroundVeryFastSoil', 'BelowGroundVeryFastSoil',
        'AboveGroundFastSoil', 'BelowGroundFastSoil', 'MediumSoil',
        'AboveGroundSlowSoil', 'BelowGroundSlowSoil', 'SoftwoodStemSnag',
        'SoftwoodBranchSnag', 'HardwoodStemSnag', 'HardwoodBranchSnag']


def get_comparison(name):
    if name == "all":
        return get_pool_mapping(
                get_cbm3_pools(),
                get_libcbm_pools())
    elif name == "biomass":
        return get_pool_mapping(
                get_cbm3_biomass_pools(),
                get_libcbm_biomass_pools())
    else:
        raise ValueError(
                "unknown comparison name, expected 'all' or 'biomass'")


def get_pool_mapping(cbm_pools, libcbm_pools):
    """returns a mapping between libcbm poolnames and cbm-cfs3 poolnames.
    Assumes the pools are ordered."""
    return collections.OrderedDict(zip(cbm_pools, libcbm_pools))


def join_pools(libCBM_pools, cbm3_pools, comparison):

    pool_mapping = get_comparison(comparison)
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
