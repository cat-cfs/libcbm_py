import os
import sys
import platform
import warnings


def get_local_dir():
    """Gets the directory containing this script

    Returns:
        str: full path to the the script's directory
    """
    return os.path.dirname(os.path.realpath(__file__))


def get_cbm_defaults_path():
    """Gets the path to the packaged cbm defaults sqlite database

    Returns:
        str: path to the bundled database
    """
    return os.path.join(
        get_local_dir(), "cbm_defaults_db", "cbm_defaults_2019.db")


def get_examples_dir():
    return os.path.join(get_local_dir(), "examples")


def get_libcbm_bin_path():
    """Returns bundled, OS specific compiled libcbm C/C++ library

    Raises:
        RuntimeError: 32 bit systems are not supported
        RuntimeError: unknown platform.system() value

    Returns:
        str: path to the bundled, compiled dll or so file
    """
    local_dir = get_local_dir()
    if sys.maxsize <= 2**32:
        raise RuntimeError("32 bit python not supported")
    system = platform.system()
    if system == "Windows":
        return os.path.join(
            local_dir, "libcbm_bin", "win_x86_64", "libcbm.dll")
    elif system == "Linux":
        plat = platform.platform()
        if "Ubuntu" not in plat or "18.04" not in plat:
            message = f"unsupported linux distribution: {plat}"
            warnings.warn(message, RuntimeWarning)
        return os.path.join(
            local_dir, "libcbm_bin", "ubuntu_18_04_x86_64", "libcbm.so")
    else:
        raise RuntimeError("unsupported platform")
