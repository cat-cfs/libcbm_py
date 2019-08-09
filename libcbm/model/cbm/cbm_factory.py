from libcbm.model.cbm import cbm_defaults
from libcbm.model.cbm.cbm_model import CBM
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


def create(dll_path, db_path, model_factory, merch_volume_to_biomass_factory,
           classifiers_factory):
    """Create and initialize an instance of the CBM model

    Args:
        dll_path (str): path to the libcbm compiled library
        db_path (str): path to a cbm_defaults formatted sqlite database
        model_factory (func): function for creating the handle to the
            low level libcbm library.  It is a function of the specified
            dll_path, and db_path that returns an initialized
            `libcbm.wrapper.libcbm_wrapper.LibCBMWrapper` instance.
        merch_volume_to_biomass_factory (func): function that creates a
            valid merchantable volume to biomass configuration for CBM
        classifiers_factory (func): function that creates a valid classifier
            configuration for CBM

    Returns:
        libcbm.model.cbm.CBM: the initialized CBM model
    """
    dll = model_factory(
        dll_path, get_cbm_defaults_configuration_factory(db_path))

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
