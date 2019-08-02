from libcbm.configuration import cbm_defaults
from libcbm.model.cbm import CBM
import json


def create(dll_path, db_path, model_factory, merch_volume_to_biomass_factory,
           classifiers_factory):
    """create and initialize an instance of the CBM model

    Arguments:
        dll_path {str} -- path to the libcbm compiled library
        db_path {str} -- path to a cbm_defaults formatted sqlite database
        model_factory {func} -- function for creating the handle to the
            low level libcbm library.
        merch_volume_to_biomass_factory {func} -- function that creates a
            valid merchantable volume to biomass configuration for CBM
        classifiers_factory {func} -- [description]

    Returns:
        libcbm.model.CBM -- instance of the CBM model
    """
    dll = model_factory(dll_path, db_path)
    merch_volume_to_biomass_config = \
        merch_volume_to_biomass_factory()
    classifiers_config = classifiers_factory()
    config = {
        "cbm_defaults": cbm_defaults.load_cbm_parameters(db_path),
        "merch_volume_to_biomass": merch_volume_to_biomass_config,
        "classifiers": classifiers_config["classifiers"],
        "classifier_values": classifiers_config["classifier_values"],
        "transitions": []
    }

    return CBM(dll, config)
