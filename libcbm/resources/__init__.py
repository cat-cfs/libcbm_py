# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys
import warnings
import platform


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
        get_local_dir(), "cbm_defaults_db", "cbm_defaults_2020.db")


def get_test_resources_dir():
    return os.path.join(get_local_dir(), "test")


def get_linux_os_release():
    path = "/etc/os-release"
    if not os.path.exists(path):
        return None
    with open(path) as fp:
        d = {}
        for line in fp:
            k, v = line.rstrip().split('=')
            if v.startswith('"'):
                v = v[1:-1]
            d[k] = v
        return d


def get_libcbm_bin_path():
    """Returns bundled, OS specific compiled libcbm C/C++ library

    Raises:
        RuntimeError: 32 bit systems are not supported
        RuntimeError: unknown platform.system() value

    Returns:
        str: path to the bundled, compiled `dll`, `so` or `dylib` file
    """
    # Get the directory that this script is located in #
    local_dir = get_local_dir()
    # Can only run on 64-bit platforms #
    if sys.maxsize <= 2**32:
        raise RuntimeError("32 bit python not supported")
    # Get the current operating system #
    # This returns a string like 'Linux', 'Darwin', 'Java' or 'Windows' #
    system = platform.system()
    # Windows case #
    if system == "Windows":
        return os.path.join(
            local_dir, "libcbm_bin", "win_x86_64", "libcbm.dll")
    # Linux case #
    elif system == "Linux":
        os_release = get_linux_os_release()
        if not os_release:
            raise RuntimeError("unsupported platform")
        version_id = os_release["VERSION_ID"]
        os_name = os_release["NAME"].lower()
        if os_name == "ubuntu" and version_id == "18.04":
            return os.path.join(
                local_dir, "libcbm_bin", "ubuntu_18_04_x86_64", "libcbm.so")
        elif os_name == "ubuntu" and version_id == "20.04":
            return os.path.join(
                local_dir, "libcbm_bin", "ubuntu_20_04_x86_64", "libcbm.so")
        elif platform.system().lower() == "linux":
            message = f"unsupported linux distribution: {platform.platform()}"
            warnings.warn(message, RuntimeWarning)
            return os.path.join(
                local_dir, "libcbm_bin", "ubuntu_18_04_x86_64", "libcbm.so")
        else:
            raise RuntimeError("unsupported platform")
    # macOS case #
    elif system == "Darwin":
        # This returns a string like '10.8.4' #
        os_release = platform.mac_ver()[0]
        # Split the result #
        major, minor, patch = os_release.split('.')
        # The directory name that contains the complied files #
        dir_name = "macosx_%s_%s_x86_64" % (major, minor)
        # Get the full path to the dylib #
        dylib = os.path.join(local_dir, "libcbm_bin", dir_name, "libcbm.dylib")
        # Let's hope we have it compiled for that version #
        msg = "The source distribution for this version of macOS has not been" \
              " compiled yet. You can do this yourself with the `libcbm_c`" \
              " repository and `cmake`."
        if not os.path.exists(dylib): raise RuntimeError(msg)
        # Otherwise return #
        return dylib
    # Other cases #
    else:
        msg = "The platform '%s' is currently unsupported."
        raise RuntimeError(msg % system)
