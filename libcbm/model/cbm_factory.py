from libcbm.configuration import cbm_defaults
from libcbm.model.cbm import CBM
import json


def create(dll_path, db_path, model_factory, merch_volume_to_biomass_factory,
           classifiers_factory):


    dll = model_factory.create(dll_path, db_path)
    merch_volume_to_biomass_config = \
        merch_volume_to_biomass_factory.create()
    classifiers_config = classifiers_factory.create()
    config = {
        "cbm_defaults": cbm_defaults.load_cbm_parameters(db_path),
        "merch_volume_to_biomass": merch_volume_to_biomass_config,
        "classifiers": classifiers_config["classifiers"],
        "classifier_values": classifiers_config["classifier_values"],
        "transitions": []
    }

    return CBM(dll, config)
