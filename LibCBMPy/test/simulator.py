def run_libCBM(dbpath, cases, nsteps, spinup_debug = False):
    
    dllpath = r'C:\dev\LibCBM\LibCBM\x64\Debug\LibCBM.dll'

    dlldir = os.path.dirname(dllpath)
    cwd = os.getcwd()
    os.chdir(dlldir)
    dll = LibCBMWrapper(dllpath)
    os.chdir(cwd)
    
    pooldef = cbm_defaults.load_cbm_pools(dbpath)
    dll.Initialize(libcbmconfig.to_string(
        {
            "pools": pooldef,
            "flux_indicators": cbm_defaults.load_flux_indicators(dbpath)
        }))
    
    #create a single classifier/classifier value for the single growth curve
    classifiers_config = cbmconfig.classifier_config([
        cbmconfig.classifier("growth_curve", [
            cbmconfig.classifier_value(get_classifier_name(c["id"])) 
            for c in cases
        ])
    ])


    transitions_config = []
    species_reference = cbm_defaults.load_species_reference(dbpath, "en-CA")
    spatial_unit_reference = cbm_defaults.get_spatial_unit_ids_by_admin_eco_name(dbpath, "en-CA")
    disturbance_types = cbm_defaults.get_disturbance_type_ids_by_name(dbpath, "en-CA")
    curves = []
    for c in cases:
        classifier_set = [get_classifier_name(c["id"])]
        merch_volumes = []
        for component in c["components"]:
            merch_volumes.append({
                "species_id": species_reference[component["species"]]["species_id"],
                "age_volume_pairs": component["age_volume_pairs"]
            })

        curve = cbmconfig.merch_volume_curve(
            classifier_set = classifier_set,
            merch_volumes = merch_volumes)
        curves.append(curve)

    merch_volume_to_biomass_config = cbmconfig.merch_volume_to_biomass_config(
        dbpath, curves)

    dll.InitializeCBM(libcbmconfig.to_string({
        "cbm_defaults": cbm_defaults.load_cbm_parameters(dbpath),
        "merch_volume_to_biomass": merch_volume_to_biomass_config,
        "classifiers": classifiers_config["classifiers"],
        "classifier_values": classifiers_config["classifier_values"],
        "transitions": []
    }))

    nstands = len(cases)
    age = np.zeros(nstands,dtype=np.int32)
    classifiers = np.zeros((nstands,1),dtype=np.int32)
    classifiers[:,0]=[classifiers_config["classifier_index"][0][get_classifier_name(c["id"])] for c in cases]
         
    spatial_units = np.array(
        [spatial_unit_reference[(c["admin_boundary"],c["eco_boundary"])]
            for c in cases],dtype=np.int32)
    pools = np.zeros((nstands,len(pooldef)))
    cbm3 = CBM3(dll)
    pool_result = pd.DataFrame()
    
    spinup_debug = cbm3.spinup(
        pools=pools,
        classifiers=classifiers,
        inventory_age=c["age"],
        spatial_unit=spatial_units,
        historic_disturbance_type=disturbance_types[c["historic_disturbance"]],
        last_pass_disturbance_type=disturbance_types[c["last_pass_disturbance"]],
        delay=c["delay"],
        mean_annual_temp=None,
        debug=spinup_debug)
    
    iteration_result = pd.DataFrame({x["name"]: pools[:,x["index"]] for x in pooldef})
    iteration_result.insert(0, "timestep", 0) 
    iteration_result.insert(0, "age", 0) #fix this, ages will be a random vector soon
    iteration_result.reset_index(level=0, inplace=True)
    pool_result = pool_result.append(iteration_result)
    
    return {"pools": pool_result, "spinup_debug": spinup_debug}