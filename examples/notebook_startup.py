import os
import sys
import json

# This is required so that the notebooks can work with the libcbm package
# directory structure.
sys.path.insert(0, os.path.abspath('..'))


def load_settings():

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
