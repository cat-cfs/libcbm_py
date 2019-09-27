import os
import json
import shutil
from types import SimpleNamespace
import pandas as pd
import win32api

from libcbm.test.cbm.cbm3_support import cbm3_simulator


def get_version_number(filename):
    """
    Gets the version number for the specified filename,
    If that filename corresponds to a .net assembly
    https://stackoverflow.com/questions/6470032/extract-assembly-version-from-dll-using-python
    """
    info = win32api.GetFileVersionInfo(filename, "\\")
    ms = info['FileVersionMS']
    ls = info['FileVersionLS']
    return ".".join([str(x) for x in [
        win32api.HIWORD(ms), win32api.LOWORD(ms),
        win32api.HIWORD(ls), win32api.LOWORD(ls)]])


def get_default_cbm3_paths():
    toolbox_install_path = os.path.join(
        "C:", "Program Files (x86)", "Operational-Scale CBM-CFS3")
    cbm_exe_dir = os.path.join(toolbox_install_path, "Admin", "executables")
    aidb_path = os.path.join(
        toolbox_install_path, "Admin", "DBs", "ArchiveIndex_Beta_Install.mdb")
    return toolbox_install_path, cbm_exe_dir, aidb_path


def load_cbm_cfs3_test(dir):
    test = SimpleNamespace()
    with open(os.path.join(dir, 'metadata.json')) as metadata_fp:
        test.metadata = json.load(metadata_fp)
    with open(os.path.join(dir, 'cases.json'), 'w') as cases_fp:
        test.cases = json.load(cases_fp)



def save_cbm_cfs3_test(name, output_dir, cbm3_project_path, cbm3_results_path,
                       age_interval, num_age_classes, n_steps, cases,
                       toolbox_install_path, cbm_exe_dir, aidb_path,
                       cbm3_result):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cbm3_project_save_path = os.path.join(
        output_dir, os.path.basename(cbm3_project_path))
    shutil.copyfile(cbm3_project_path, cbm3_project_save_path)

    cbm3_results_save_path = os.path.join(
        output_dir, os.path.basename(cbm3_results_path)
    )
    shutil.copyfile(cbm3_results_path, cbm3_results_save_path)

    metadata = {
        "name": name,
        "cbm3_project_path": cbm3_project_save_path,
        "cbm3_results_path": cbm3_results_save_path,
        "age_interval": age_interval,
        "num_age_classes": num_age_classes,
        "n_steps": n_steps,
        "toolbox_install_path": toolbox_install_path,
        "toolbox_version": get_version_number(
            os.path.join(toolbox_install_path, "Toolbox.exe")),
        "cbm_exe_dir": cbm_exe_dir,
        "aidb_path": aidb_path
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as metadata_fp:
        json.dump(metadata, metadata_fp)
    with open(os.path.join(output_dir, 'cases.json'), 'w') as cases_fp:
        json.dump(cases, cases_fp)

    cbm3_result["pools"].to_csv(os.path.join(output_dir, "pools.csv"))
    cbm3_result["flux"].to_csv(os.path.join(output_dir, "flux.csv"))
    cbm3_result["state"].to_csv(os.path.join(output_dir, "state.csv"))
    return metadata


def run_cases_cbm_cfs3(name, output_dir, cases, age_interval, num_age_classes,
                       n_steps):
    toolbox_install_path, cbm3_exe_path, archive_index_db_path \
        = get_default_cbm3_paths()

    cbm3_project_path = cbm3_simulator.get_project_path(
        toolbox_install_path, name)
    sit_config_save_path = cbm3_simulator.get_config_path(
        toolbox_install_path, name)
    project_path = cbm3_simulator.import_cbm3_project(
        name="stand_level_testing",
        cases=cases,
        age_interval=age_interval,
        num_age_classes=num_age_classes,
        n_steps=n_steps,
        toolbox_path=toolbox_install_path,
        archive_index_db_path=archive_index_db_path,
        sit_config_save_path=sit_config_save_path,
        cbm3_project_path=cbm3_project_path)

    cbm3_results_path = cbm3_simulator.run_cbm3(
        archive_index_db_path=archive_index_db_path,
        project_path=project_path,
        toolbox_path=toolbox_install_path,
        cbm_exe_path=cbm3_exe_path)

    cbm3_result = cbm3_simulator.get_cbm3_results(cbm3_results_path)
    save_cbm_cfs3_test(
        name, output_dir, cbm3_project_path, cbm3_results_path,
        age_interval, num_age_classes, n_steps, cases, toolbox_install_path,
        cbm3_exe_path, archive_index_db_path, cbm3_result)