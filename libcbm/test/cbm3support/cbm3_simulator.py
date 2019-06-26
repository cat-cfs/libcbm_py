from libcbm.test.cbm3support import cbm3_python_helper
cbm3_python_helper.load_cbm3_python()
from cbm3_python.simulation import projectsimulator
from cbm3_python.cbm3data import sit_helper
from cbm3_python.cbm3data import cbm3_results
from libcbm.test import casegeneration
import os

def get_unfccc_land_class_id_ref():
    return { "UNFCCC_FL_R_FL": 0, "UNFCCC_CL_R_CL": 1, "UNFCCC_GL_R_GL": 2,
    "UNFCCC_WL_R_WL": 3, "UNFCCC_SL_R_SL": 4, "UNFCCC_OL_R_OL": 5,
    "UNFCCC_CL_R_FL": 6, "UNFCCC_GL_R_FL": 7, "UNFCCC_WL_R_FL": 8,
    "UNFCCC_SL_R_FL": 9, "UNFCCC_OL_R_FL": 10, "UNFCCC_FL_R_CL": 11,
    "UNFCCC_FL_R_GL": 12, "UNFCCC_FL_R_WL": 13, "UNFCCC_FL_R_SL": 14,
    "UNFCCC_FL_R_OL": 15, "UNFCCC_UFL": 16, "UNFCCC_UFL_R_CL": 17,
    "UNFCCC_UFL_R_GL": 18, "UNFCCC_UFL_R_WL": 19, "UNFCCC_UFL_R_SL": 20,
    "UNFCCC_UFL_R_OL": 21, "UNFCCC_UFL_R_FL": 22, "PEATLAND": 13 }

def get_project_path(toolbox_path, name):
    #creates a default path for a cbm project databases in the toolbox installation dir
    return os.path.join(toolbox_path, "Projects", name, "{}.mdb".format(name))


def get_results_path(project_path):
    #creates a default path for a cbm results database in the toolbox installation dir
    name = os.path.splitext(os.path.basename(project_path))[0]
    return os.path.join(os.path.dirname(project_path), "{}_results.mdb".format(name))


def get_config_path(toolbox_path, name):
    #creates a default path for a saving the SIT configuration
    cbm3_project_dir = os.path.dirname(get_project_path(toolbox_path, name))
    return os.path.join(cbm3_project_dir, "{}.json".format(name))


def import_cbm3_project(name, cases, age_interval, num_age_classes, nsteps, cbm_exe_path,
        toolbox_path, archive_index_db_path, cbm3_project_path=None, sit_config_save_path=None):

    standard_import_tool_plugin_path=sit_helper.load_standard_import_tool_plugin()

    #there is a bug fix in this version of cbm/makelist for growth increment blips
    #cbm_exe_path = r"M:\CBM Tools and Development\Builds\CBMBuilds\20190530_growth_increment_fix"

    if not cbm3_project_path:
        cbm3_project_path = get_project_path(toolbox_path, name)
    if not sit_config_save_path:
        sit_config_save_path = get_config_path(toolbox_path, name)

    sit_config = sit_helper.SITConfig(
        imported_project_path=cbm3_project_path,
        initialize_mapping=True,
        archive_index_db_path=archive_index_db_path
    )
    sit_config.data_config(
        age_class_size=age_interval,
        num_age_classes=num_age_classes,
        classifiers=["admin", "eco", "identifier", "species"])
    sit_config.set_admin_eco_mapping("admin","eco")
    sit_config.set_species_classifier("species")
    for c in cases:
        species = None
        if not c["afforestation_pre_type"] is None:
            species = c["afforestation_pre_type"]
        else:
            species = "Spruce" #"Spruce" does not actually matter here, since ultimately species composition is decided in yields
        cset = [
            c["admin_boundary"],
            c["eco_boundary"],
            casegeneration.get_classifier_name(c["id"]),
            species]
        sit_config.add_inventory(classifier_set=cset, area=c["area"],
            age=c["age"], unfccc_land_class=get_unfccc_land_class_id_ref()[c["unfccc_land_class"]],
            delay=c["delay"], historic_disturbance=c["historic_disturbance"],
            last_pass_disturbance=c["last_pass_disturbance"])
        for component in c["components"]:
            sit_config.add_yield(classifier_set=cset, 
                        leading_species_classifier_value=component["species"],
                        values=[x[1] for x in component["age_volume_pairs"]])
        for event in c["events"]:
            sit_config.add_event(
                classifier_set=cset,
                disturbance_type=event["disturbance_type"],
                time_step=event["time_step"],
                target=1, # not yet supporting disturbance rules here, meaning each event will target only a single stand
                target_type = "Area",
                sort = "SORT_BY_SW_AGE")
    sit_config.add_event(
        classifier_set=["?","?","?","?"],
        disturbance_type="Wildfire",
        time_step=nsteps+1,
        target=1,
        target_type = "Area",
        sort = "SORT_BY_SW_AGE")
    sit_config.import_project(standard_import_tool_plugin_path,
        sit_config_save_path)
    return cbm3_project_path


def run_cbm3(aidb_path, project_path, toolbox_path, cbm_exe_path,
    cbm3_results_db_path=None):
    if not cbm3_results_db_path:
        cbm3_results_db_path = get_results_path(project_path)
    projectsimulator.run(
        aidb_path=aidb_path,
        project_path=project_path,
        toolbox_installation_dir=toolbox_path,
        cbm_exe_path=cbm_exe_path,
        results_database_path = cbm3_results_db_path)
    return cbm3_results_db_path

def get_cbm3_results(cbm3_results_db_path):
    cbm3_pool_result = cbm3_results.load_pool_indicators(cbm3_results_db_path, classifier_set_grouping=True)
    cbm3_flux_result = cbm3_results.load_stock_changes(cbm3_results_db_path, classifier_set_grouping=True)
    return {
        "pools": cbm3_pool_result,
        "flux": cbm3_flux_result
        }