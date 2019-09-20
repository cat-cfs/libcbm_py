import os
import sys
import platform


def get_local_dir():
    return os.path.dirname(os.path.realpath(__file__))


def get_cbm_defaults_path():
    return os.path.join(
        get_local_dir(), "cbm_defaults_db", "cbm_defaults_2019.db")


def get_libcbm_bin_path():
    local_dir = get_local_dir()
    if sys.maxsize <= 2**32:
        raise RuntimeError("32 bit python not supported")
    system = platform.system()
    if system == "Windows":
        return os.path.join(
            local_dir, "libcbm_bin", "win_x86_64", "libcbm.dll")
    elif system == "Linux":
        plat = platform.platform()
        if "Ubuntu" in plat and "18.04" in plat:
            raise RuntimeError("unsupported linux distribution")
        return os.path.join(
            local_dir, "libcbm_bin", "ubuntu_18_04_x86_64", "libcbm.so")
    else:
        raise RuntimeError("unsupported platform")
