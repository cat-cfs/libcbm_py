import os
import sys
import json

# This is required so that the notebooks can work with the libcbm package
# directory structure.
package_dir = os.path.abspath('..')
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)


def load_settings():
    """Load or create a settings file which is a json dictionary with the
    following string variables:

        - cbm3_exe_path: directory containing the CBM-CFS3
                         model executables: cbm.exe and makelist.exe
        - toolbox_path: directory of the CBM-CFS3 application installation
        - archive_index_db_path: path to a CBM-CFS3 archive index database
        - cbm_defaults_db_path: path to a cbm_defaults database. See:
                                https://github.com/cat-cfs/cbm_defaults
        - libcbm_path: path to a compiled libcbm dll/so file.

    If the file does not already exist an empty one is created.

    Returns:
        dict: a dictionary with settings
    """
    local_dir = os.path.dirname(os.path.realpath(__file__))
    local_settings_path = os.path.join(
        local_dir, "notebook_settings.json")
    default_settings = {
        "cbm3_exe_path": None,
        "toolbox_path": None,
        "archive_index_db_path": None,
        "cbm_defaults_db_path": None,
        "libcbm_path": None
    }
    if not os.path.exists(local_settings_path):
        with open(local_settings_path, 'w') as local_settings_file:
            json.dump(
                default_settings,
                local_settings_file,
                indent=4)
        print("local settings file '{0}' created. "
              .format(local_settings_path) +
              "Please set your paths in this file.")
        return default_settings
    else:
        with open(local_settings_path) as local_settings_file:
            settings = json.load(local_settings_file)
            return settings
