def join_pools():
    
    libCBM_pools = libCBM_pool_result[['identifier','timestep', 'SoftwoodMerch', 'SoftwoodFoliage', 
                   'SoftwoodOther', 'SoftwoodCoarseRoots', 'SoftwoodFineRoots',
                   'HardwoodMerch', 'HardwoodFoliage', 'HardwoodOther',
                   'HardwoodCoarseRoots', 'HardwoodFineRoots', 'AboveGroundVeryFastSoil',
                   'BelowGroundVeryFastSoil', 'AboveGroundFastSoil', 'BelowGroundFastSoil', 'MediumSoil',
                   'AboveGroundSlowSoil', 'BelowGroundSlowSoil', 'SoftwoodStemSnag', 'SoftwoodBranchSnag',
                   'HardwoodStemSnag', 'HardwoodBranchSnag']]

    cbm3_pools = cbm3_pool_result[['identifier','TimeStep', 'Softwood Merchantable', 'Softwood Foliage',
                 'Softwood Other', 'Softwood Coarse Roots', 'Softwood Fine Roots',
                 'Hardwood Merchantable', 'Hardwood Foliage', 'Hardwood Other',
                 'Hardwood Coarse Roots', 'Hardwood Fine Roots', 'Aboveground Very Fast DOM',
                 'Belowground Very Fast DOM', 'Aboveground Fast DOM', 'Belowground Fast DOM', 'Medium DOM',
                 'Aboveground Slow DOM', 'Belowground Slow DOM', 'Softwood Stem Snag', 'Softwood Branch Snag',
                 'Hardwood Stem Snag', 'Hardwood Branch Snag']]

    #make column naming consistent
    cbm3_pools = cbm3_pools.rename(columns = {
        'Softwood Merchantable': 'SoftwoodMerch',
        'Softwood Foliage': 'SoftwoodFoliage',
        'Softwood Other': 'SoftwoodOther',
        'Softwood Coarse Roots': 'SoftwoodCoarseRoots',
        'Softwood Fine Roots': 'SoftwoodFineRoots',
        'Hardwood Merchantable': 'HardwoodMerch',
        'Hardwood Foliage': 'HardwoodFoliage',
        'Hardwood Other': 'HardwoodOther',
        'Hardwood Coarse Roots': 'HardwoodCoarseRoots',
        'Hardwood Fine Roots': 'HardwoodFineRoots',
        'Aboveground Very Fast DOM': "AboveGroundVeryFastSoil",
        'Belowground Very Fast DOM': "BelowGroundVeryFastSoil",
        'Aboveground Fast DOM': "AboveGroundFastSoil",
        'Belowground Fast DOM': "BelowGroundFastSoil",
        'Medium DOM': "MediumSoil",
        'Aboveground Slow DOM': "AboveGroundSlowSoil",
        'Belowground Slow DOM': "BelowGroundSlowSoil",
        'Softwood Stem Snag': "SoftwoodStemSnag",
        'Softwood Branch Snag': "SoftwoodBranchSnag",
        'Hardwood Stem Snag': "HardwoodStemSnag",
        'Hardwood Branch Snag': "HardwoodBranchSnag"})

    merged = libCBM_pools.merge(cbm3_pools,
                                  left_on=['identifier','timestep'],
                                  right_on=['identifier','TimeStep'],
                                  suffixes=("_libCBM","_cbm3"))

    diff_colnames = []
    cbm3_colnames = []
    libCBM_colnames = []
    merged["total_diff"]=0
    #compute diffs row-by-row
    for pool in ['SoftwoodMerch', 'SoftwoodFoliage', 'SoftwoodOther', 'SoftwoodCoarseRoots',
                 'SoftwoodFineRoots','HardwoodMerch', 'HardwoodFoliage', 'HardwoodOther',
                 'HardwoodCoarseRoots', 'HardwoodFineRoots','AboveGroundVeryFastSoil',
                 'BelowGroundVeryFastSoil', 'AboveGroundFastSoil', 'BelowGroundFastSoil', 'MediumSoil',
                 'AboveGroundSlowSoil', 'BelowGroundSlowSoil', 'SoftwoodStemSnag', 'SoftwoodBranchSnag',
                 'HardwoodStemSnag', 'HardwoodBranchSnag']:
    
        d = "{}_diff".format(pool)
        l = "{}_libCBM".format(pool)
        r = "{}_cbm3".format(pool)
        diff_colnames.append(d)
        cbm3_colnames.append(r)
        libCBM_colnames.append(l)
        merged[d] = (merged[l] - merged[r])
        merged["total_diff"]+=merged[d].abs()
    merged["total_diff"] = merged["total_diff"]
    return merged