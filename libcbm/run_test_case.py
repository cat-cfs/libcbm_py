case = {
 'id': 4,
 'age': 0,
 'area': 1.0,
 'delay': 0,
 'unfccc_land_class': 0,
 'admin_boundary': 'Quebec',
 'eco_boundary': 'Hudson Plains',
 'historic_disturbance': 'Wildfire',
 'last_pass_disturbance': 'Wildfire',
 'components': [{'species': 'Upland hardwoods other than sugar maple',
   'age_volume_pairs': [(0, 30.07),
    (5, 46.07),
    (10, 68.02),
    (15, 95.66),
    (20, 126.96),
    (25, 158.4),
    (30, 186.39),
    (35, 208.76),
    (40, 225.16),
    (45, 236.42),
    (50, 243.81),
    (55, 248.53),
    (60, 251.48),
    (65, 253.3),
    (70, 254.42),
    (75, 255.1),
    (80, 255.52),
    (85, 255.77),
    (90, 255.93),
    (95, 256.02),
    (100, 256.08),
    (105, 256.11),
    (110, 256.13),
    (115, 256.14),
    (120, 256.15),
    (125, 256.16),
    (130, 256.16),
    (135, 256.16),
    (140, 256.16),
    (145, 256.16),
    (150, 256.16),
    (155, 256.16),
    (160, 256.16),
    (165, 256.16),
    (170, 256.16),
    (175, 256.16),
    (180, 256.16),
    (185, 256.16),
    (190, 256.16),
    (195, 256.16)]}],
 'events': [{'disturbance_type': 'Afforestation', 'time_step': 247}]}

age_interval=5
num_age_classes = 40 #required by cbm3
n_steps = 250
cbm3_exe_path = r"M:\CBM Tools and Development\Builds\CBMBuilds\20190530_growth_increment_fix"
toolbox_path = r"C:\Program Files (x86)\Operational-Scale CBM-CFS3"
archive_index_db_path = r"C:\Program Files (x86)\Operational-Scale CBM-CFS3\Admin\DBs\ArchiveIndex_Beta_Install.mdb"

cbm_defaults_db_path = 'C:\dev\cbm_defaults\cbm_defaults.db'
libcbm_path = r'C:\dev\LibCBM\LibCBM\x64\Debug\LibCBM.dll'

from libcbm.test import simulator
libcbm_result = simulator.run_libCBM(libcbm_path, cbm_defaults_db_path, [case], n_steps, spinup_debug=True)

#from libcbm.test.cbm3support import cbm3_simulator
#project_path = cbm3_simulator.import_cbm3_project(
#    name="stand_level_testing",
#    cases=[case],
#    age_interval=age_interval,
#    num_age_classes=num_age_classes,
#    nsteps=n_steps,
#    cbm_exe_path=cbm3_exe_path,
#    toolbox_path=toolbox_path,
#    archive_index_db_path=archive_index_db_path)
#
#cbm3_results_path = cbm3_simulator.run_cbm3(
#    aidb_path=archive_index_db_path, 
#    project_path=project_path,
#    toolbox_path=toolbox_path,
#    cbm_exe_path=cbm3_exe_path)
#
#cbm3_result = cbm3_simulator.get_cbm3_results(cbm3_results_path)