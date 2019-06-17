
def initialize_config( dbpath, classifiers, transitions,
                      merch_volume_to_biomass=None,
                      save_path=None):
    '''
    initialize config sets up the json configuration object passed to the underlying dll
    returns config as string, and optionally saves to specified path
    '''
    configuration = {}
    configuration["cbm_defaults"] = cbm_defaults.load_cbm_parameters(dbpath)
    configuration["merch_volume_to_biomass"] = merch_volume_to_biomass
    configuration["classifiers"] = classifiers["classifiers"]
    configuration["classifier_values"] = classifiers["classifier_values"]
    configuration["transitions"] = transitions
    if save_path:
        save_config(save_path, configuration)

    return configuration


def classifier_value(value, description=""):
    return {"id": None, "classifier_id": None, "value": value, "description": description }

def classifier(name, values):
    return {"id": None, "name": name },values

def classifier_config(classifiers):
    result = {
        "classifiers": [],
        "classifier_values": [],
        "classifier_index": []
    }
    for i, c in enumerate(classifiers):
        classifier = c[0]
        values = c[1]
        classifier["id"] = i + 1
        result["classifiers"].append(classifier)
        index  = {}
        for j,cv in enumerate(values):
            cv["id"] = j+1
            cv["classifier_id"] = classifier["id"]
            result["classifier_values"].append(cv)
            index[cv["value"]] = cv["id"]
        result["classifier_index"].append(index)
    return result

def merch_volume_curve(classifier_set, merch_volumes):
    result = {
        "classifier_set": { "type": "name", "values": [x for x in classifier_set] },
    }
    components = []
    for m in merch_volumes:
        components.append({
            "species_id": m["species_id"],
            "age_volume_pairs": m["age_volume_pairs"]
            })
    result["components"] = components
    return result

def merch_volume_to_biomass_config(dbpath, merch_volume_curves):
    return {"db_path": dbpath, "merch_volume_curves": merch_volume_curves }


def initialize_merch_volume_to_biomass_config(dbpath, yield_table_path,
    yield_age_class_size, yield_table_header_row, classifiers_config):
    yield_table_config =  sityield.read_sit_yield(
        yield_table_path, dbpath, classifiers_config, yield_age_class_size,
        header=yield_table_header_row)
    return { "db_path": dbpath, "merch_volume_curves": yield_table_config }






