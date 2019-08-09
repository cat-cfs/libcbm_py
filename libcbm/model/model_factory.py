
from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
import json


def create(dll_path, configuration_factory):
    """Creates and initializes a handle to the low level libcbm library.

    Args:
        dll_path (str): path to the libcbm compiled library
        configuration_factory (func): Parameterless function that returns
            a libcbm pool configuration. See:
            :py:func:`libcbm.wrapper.libcbm_wrapper.LibCBMWrapper.Initialize`
            for the expected format of this configuration.

    Returns:
        LibCBMWrapper: class with python wrapper functions for low level
            library
    """

    dll = LibCBMWrapper(dll_path)
    dll_config = configuration_factory()
    dll_config_string = json.dumps(dll_config)
    dll.Initialize(dll_config_string)
    return dll
