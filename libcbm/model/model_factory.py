from libcbm.configuration import cbm_defaults
from libcbm.wrapper.libcbmwrapper import LibCBMWrapper
import json


def create(dll_path, db_path):
    """Creates and initializes a handle to the low level libcbm library.

    Arguments:
        dll_path {str} -- path to the libcbm compiled library
        db_path {str} -- path to a cbm_defaults formatted sqlite database

    Returns:
        LibCBMWrapper -- class with python wrapper functions for low level
            library
    """
    pooldef = cbm_defaults.load_cbm_pools(db_path)
    flux_ind = cbm_defaults.load_flux_indicators(db_path)
    dll = LibCBMWrapper(dll_path)
    dll_config = {
            "pools": pooldef,
            "flux_indicators": flux_ind
        }
    dll_config_string = json.dumps(dll_config)
    dll.Initialize(dll_config_string)
    return dll
