import json
import cbm_defaults
import sityield

def loadjson(path):
    with open(path) as f:
        data = json.load(f)
    return data

def initialize_config( dbpath, classifiers, transitions,
                      merch_volume_to_biomass=None,
                      save_path=None):
    '''
    initialize config sets up the json configuration object passed to the underlying dll
    returns config as string, and optionally saves to specified path
    '''
    configuration = {}
    configuration["cbm_defaults"] = cbm_defaults.load_cbm_parameters(dbpath)
    configuration["pools"] = cbm_defaults.load_cbm_pools(dbpath)
    configuration["flux_indicators"] = cbm_defaults.load_flux_indicators(dbpath)
    configuration["merch_volume_to_biomass"] = merch_volume_to_biomass
    configuration["classifiers"] = classifiers["classifiers"]
    configuration["classifier_values"] = classifiers["classifier_values"]
    configuration["transitions"] = transitions
    configString = json.dumps(configuration, indent=4)#, ensure_ascii=True)
    if save_path:
        with open(save_path, 'w') as configfile:
            configfile.write(configString)
    return configString

def initialize_merch_volume_to_biomass_config(dbpath, yield_table_path,
    yield_age_class_size, yield_table_header_row, classifiers_config):
    yield_table_config =  sityield.read_sit_yield(
        yield_table_path, dbpath, classifiers_config, yield_age_class_size,
        header=yield_table_header_row)
    return { "db_path": dbpath, "merch_volume_curves": yield_table_config }