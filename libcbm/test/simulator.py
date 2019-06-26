import numpy as np
import pandas as pd
from libcbm.configuration import cbm_defaults
from libcbm.wrapper.libcbmwrapper import LibCBMWrapper
from libcbm.model.cbm import CBM
from libcbm.configuration import libcbmconfig
from libcbm.configuration import cbmconfig
from libcbm.test import casegeneration

def append_pools_data(df, nstands, timestep, pools, pooldef):
    data = {"timestep": timestep, "identifier": [casegeneration.get_classifier_name(x) for x in range(1,nstands+1)]}
    data.update({x["name"]: pools[:,x["index"]] for x in pooldef})
    cols=["timestep","identifier"] + [x["name"] for x in pooldef]
    df = df.append(pd.DataFrame(data=data, columns=cols))
    return df

def run_libCBM(dllpath, dbpath, cases, nsteps, spinup_debug = False):

    dll = LibCBMWrapper(dllpath)

    pooldef = cbm_defaults.load_cbm_pools(dbpath)
    flux_ind = cbm_defaults.load_flux_indicators(dbpath)
    dll.Initialize(libcbmconfig.to_string(
        {
            "pools": pooldef,
            "flux_indicators": flux_ind
        }))

    #create a single classifier/classifier value for the single growth curve
    classifiers_config = cbmconfig.classifier_config([
        cbmconfig.classifier("growth_curve", [
            cbmconfig.classifier_value(casegeneration.get_classifier_name(c["id"]))
            for c in cases
        ])
    ])

    transitions_config = []
    species_reference = cbm_defaults.load_species_reference(dbpath, "en-CA")
    spatial_unit_reference = cbm_defaults.get_spatial_unit_ids_by_admin_eco_name(dbpath, "en-CA")
    disturbance_types_reference = cbm_defaults.get_disturbance_type_ids_by_name(dbpath, "en-CA")
    afforestation_pre_types = cbm_defaults.get_afforestation_types_by_name(dbpath, "en-CA")
    land_class_ref = cbm_defaults.get_land_class_reference(dbpath, "en-CA")
    land_classes_by_code = {x["land_class_code"]: x for x in land_class_ref}
    
    
    curves = []
    for c in cases:
        classifier_set = [casegeneration.get_classifier_name(c["id"])]
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

    inventory_age = np.array([c["age"] for c in cases], dtype=np.int32)

    historic_disturbance_type = np.array(
        [disturbance_types_reference[c["historic_disturbance"]]
            for c in cases], dtype=np.int32)
    last_pass_disturbance_type = np.array(
        [disturbance_types_reference[c["last_pass_disturbance"]]
            for c in cases], dtype=np.int32)

    delay = np.array([c["delay"] for c in cases], dtype=np.int32)

    classifiers = np.zeros((nstands,1),dtype=np.int32)
    classifiers[:,0]=[classifiers_config["classifier_index"][0] \
        [casegeneration.get_classifier_name(c["id"])] for c in cases]

    spatial_units = np.array(
        [spatial_unit_reference[(c["admin_boundary"],c["eco_boundary"])]
            for c in cases],dtype=np.int32)

    afforestation_pre_type_ids = []
    for c in cases:
        if not c["afforestation_pre_type"] is None:
            afforestation_pre_type_ids.append(
                afforestation_pre_types[c["afforestation_pre_type"]])
        else:
            afforestation_pre_type_ids.append(0)

    afforestation_pre_type_id = np.array(afforestation_pre_type_ids, dtype=np.int32)

    land_class = np.ones(nstands, dtype=np.int32)
    last_disturbance_type = np.zeros(nstands, dtype=np.int32)
    time_since_last_disturbance = np.zeros(nstands, dtype=np.int32)
    time_since_land_class_change = np.zeros(nstands, dtype=np.int32)
    growth_enabled = np.zeros(nstands, dtype=np.int32)

    age = np.zeros(nstands, dtype=np.int32)
    growth_multipliers = np.ones(nstands, dtype=np.float)
    regeneration_delay = np.zeros(nstands, dtype=np.int32)
    disturbance_types = np.zeros(nstands, dtype=np.int32)
    transition_rules = np.zeros(nstands, dtype=np.int32)

    pools = np.zeros((nstands,len(pooldef)))
    flux = np.zeros((nstands, len(flux_ind)))
    enabled = np.ones(nstands, dtype=np.int32)

    disturbances = {}
    for i_c, c in enumerate(cases):
        for e in c["events"]:
            time_step = e["time_step"]
            dist_type_id = disturbance_types_reference[e["disturbance_type"]]
            if i_c in disturbances:
                if time_step in disturbances[i_c]:
                    raise ValueError("more than one event found for index {0}, timestep {1}"
                                     .format(i_c, time_step))
                else:
                    disturbances[i_c][time_step] = dist_type_id
            else:
                disturbances[i_c] = {time_step: dist_type_id}


    cbm3 = CBM(dll)
    pool_result = pd.DataFrame()

    spinup_debug = cbm3.spinup(
        pools=pools,
        classifiers=classifiers,
        inventory_age=inventory_age,
        spatial_unit=spatial_units,
        afforestation_pre_type_id=afforestation_pre_type_id,
        historic_disturbance_type=historic_disturbance_type,
        last_pass_disturbance_type=last_pass_disturbance_type,
        delay=delay,
        mean_annual_temp=None,
        debug=spinup_debug)

    cbm3.init(
        last_pass_disturbance_type=last_pass_disturbance_type,
        delay=delay,
        inventory_age=inventory_age,
        spatial_unit=spatial_units,
        afforestation_pre_type_id=afforestation_pre_type_id,
        pools=pools,
        last_disturbance_type=last_disturbance_type,
        time_since_last_disturbance=time_since_last_disturbance,
        time_since_land_class_change=time_since_land_class_change,
        growth_enabled=growth_enabled,
        enabled=enabled,
        land_class=land_class,
        age=age)

    pool_result = append_pools_data(pool_result, nstands, 0, pools, pooldef)

    for t in range(1, nsteps+1):

        disturbance_types = disturbance_types * 0
        for k,v in disturbances.items():
            if t in v:
                disturbance_types[k] = v[t]

        cbm3.step(
            pools=pools,
            flux=flux,
            classifiers=classifiers,
            age=age,
            disturbance_types = disturbance_types,
            spatial_unit=spatial_units,
            mean_annual_temp=None,
            transition_rule_ids=transition_rules,
            last_disturbance_type=last_disturbance_type,
            time_since_last_disturbance=time_since_last_disturbance,
            time_since_land_class_change=time_since_last_disturbance,
            growth_enabled=growth_enabled,
            enabled=enabled,
            land_class=land_class,
            growth_multipliers=growth_multipliers,
            regeneration_delay=regeneration_delay)
        pool_result = append_pools_data(pool_result, nstands, t, pools, pooldef)

    return {
        "pools": pool_result,
        "spinup_debug": spinup_debug
    }