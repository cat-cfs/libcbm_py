# Modules #
import os
import sys
import json
import inspect

# Get the current directory  #
file_name = inspect.getframeinfo(inspect.currentframe()).filename
this_dir  = os.path.dirname(os.path.abspath(file_name)) + '/'

# Check that the libcbm package is in the python path
# For instance if the user didn't put it in his .bash_profile
package_dir = os.path.abspath(this_dir + '../')
if package_dir not in sys.path: sys.path.insert(0, package_dir)

# Constants #
default_settings = {
    "cbm3_exe_path":         None,
    "toolbox_path":          None,
    "archive_index_db_path": None,
    "cbm_defaults_db_path":  None,
    "libcbm_path":           None
}

###############################################################################
def load_settings():
    """Load or create a settings file which is a json dictionary with the
    following string variables:

        - libcbm_path: path to a compiled libcbm dll/so file.
        - cbm_defaults_db_path: path to a cbm_defaults database. See:
                        https://github.com/cat-cfs/cbm_defaults

        The following are used only when comparing with the old CBM:

        - cbm3_exe_path: directory containing the CBM-CFS3
                         model executables: cbm.exe and makelist.exe
        - toolbox_path: directory of the CBM-CFS3 application installation
        - archive_index_db_path: path to a CBM-CFS3 archive index database

    If the file does not already exist an empty one is created.

    Returns:
        dict: a dictionary with settings
    """
    # The settings file is always located here #
    local_settings_path = os.path.join(this_dir, "notebook_settings.json")
    # Create it if it doesn't exist #
    if not os.path.exists(local_settings_path):
        with open(local_settings_path, 'w') as handle:
            json.dump(default_settings, handle, indent=4)
        print(f"Local settings file '{local_settings_path}' created."
              " Please set your paths in this file.")
    # Now read the file #
    with open(local_settings_path) as handle:
        settings = json.load(handle)
        for key in settings:
            settings[key] = os.path.normpath(settings[key])
        return settings
