# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys
import warnings
import platform
import configparser
import itertools


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
        get_local_dir(), "cbm_defaults_db", "cbm_defaults_v1.2.8340.362.db"
    )


def get_cbm_exn_parameters_dir():
    """
    gets a path to a directory containing default parameters for
    :py:mod:`libcbm.cbm_exn.cbm_exn_model`

    Returns:
        str: cbm_exn parameters
    """
    return os.path.join(get_local_dir(), "cbm_exn")


def get_test_resources_dir():
    """
    gets a path to a directory containing files for integration testing
    and examples

    Returns:
        str: test resources path
    """
    return os.path.join(get_local_dir(), "test")


def parse_key_value_file(path):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    with open(path) as fp:
        cfg.read_file(itertools.chain(["[global]", os.linesep], fp), source=path)
        return {k: v.strip("\"'") for k, v in cfg["global"].items()}


def get_linux_os_release():
    path = "/etc/os-release"
    if not os.path.exists(path):
        return None
    return parse_key_value_file(path)


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
        return os.path.join(local_dir, "libcbm_bin", "win_x86_64", "libcbm.dll")
    # Linux case #
    elif system == "Linux":
        os_release = get_linux_os_release()
        if not os_release:
            raise RuntimeError("unsupported platform")
        version_id = os_release["VERSION_ID"]
        os_name = os_release["NAME"].lower()
        if os_name == "ubuntu" and version_id == "22.04":
            return os.path.join(
                local_dir, "libcbm_bin", "ubuntu_22_04_x86_64", "libcbm.so"
            )
        elif os_name == "ubuntu" and version_id == "24.04":
            return os.path.join(
                local_dir, "libcbm_bin", "ubuntu_24_04_x86_64", "libcbm.so"
            )
        else:
            message = f"untested linux distribution: {os_release}"
            warnings.warn(message, RuntimeWarning)
            return os.path.join(
                local_dir, "libcbm_bin", "ubuntu_22_04_x86_64", "libcbm.so"
            )
    # macOS case #
    elif system == "Darwin":
        # This returns a string like '10.8.4' #
        os_release = platform.mac_ver()[0]
        # Split the result #
        version_tokens = os_release.split(".")
        major = version_tokens[0]
        minor = version_tokens[1]
        matched_ver = (int(major) == 10 and int(minor) >= 12) or (
            int(major) >= 11
        )
        msg = ("The source distribution for this version of macOS has not been compiled yet. "
                "You can do this yourself with the `libcbm_c` repository and `cmake`.")
        if not matched_ver:
            raise RuntimeError(msg)
        # Process architecture
        # Apple Silicon Python reports 'arm64'. Intel and Rosetta Python report 'x86_64'.
        machine = platform.machine().lower()
        if "arm64" in machine:
            arch_dir = "macos-arm64"
        elif "x86_64" in machine:
            arch_dir = "macos_x86_64"
        else:
            raise RuntimeError(msg)
        
        candidates = [os.path.join(local_dir, "libcbm_bin", arch_dir, "libcbm.dylib")]

        for dylib in candidates:
            if os.path.exists(dylib):
                return dylib
        
        raise RuntimeError(msg)
    # Other cases #
    else:
        msg = "The platform '%s' is currently unsupported."
        raise RuntimeError(msg % system)
