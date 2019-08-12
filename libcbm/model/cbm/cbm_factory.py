from libcbm.model.cbm import cbm_defaults
from libcbm.model.cbm.cbm_model import CBM
from libcbm.wrapper.cbm.cbm_wrapper import CBMWrapper
from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.libcbm_handle import LibCBMHandle
import json


def get_cbm_defaults_configuration_factory(db_path):
    """Get a parameterless function that creates configuration for
    :py:class:`libcbm.wrapper.libcbm_wrapper.LibCBMWrapper`

    Args:
        db_path (str): path to a cbm_defaults database

    Returns:
        func: a function that creates CBM configuration input for libcbm
    """
    def factory():
        return {
            "pools": cbm_defaults.load_cbm_pools(db_path),
            "flux_indicators": cbm_defaults.load_cbm_flux_indicators(db_path)
        }
    return factory


def create(dll_path, db_path, merch_volume_to_biomass_factory,
           classifiers_factory):
    """Create and initialize an instance of the CBM model

    Args:
        dll_path (str): path to the libcbm compiled library
        db_path (str): path to a cbm_defaults formatted sqlite database
        merch_volume_to_biomass_factory (func): function that creates a
            valid merchantable volume to biomass configuration for CBM
        classifiers_factory (func): function that creates a valid classifier
            configuration for CBM

    Returns:
        libcbm.model.cbm.CBM: the initialized CBM model
    """

    configuration_factory = get_cbm_defaults_configuration_factory(db_path)
    configuration_string = json.dumps(configuration_factory())
    libcbm_handle = LibCBMHandle(dll_path, configuration_string)
    libcbm_wrapper = LibCBMWrapper(libcbm_handle)

    merch_volume_to_biomass_config = \
        merch_volume_to_biomass_factory()
    classifiers_config = classifiers_factory()
    cbm_config = {
        "cbm_defaults": cbm_defaults.load_cbm_parameters(db_path),
        "merch_volume_to_biomass": merch_volume_to_biomass_config,
        "classifiers": classifiers_config["classifiers"],
        "classifier_values": classifiers_config["classifier_values"],
        "transitions": []
    }
    cbm_config_string = json.dumps(cbm_config)
    cbm_wrapper = CBMWrapper(libcbm_handle, cbm_config_string)
    return CBM(libcbm_wrapper, cbm_wrapper)
