import unittest
from types import SimpleNamespace
import pandas as pd
from libcbm.input.sit import sit_cbm_factory
from libcbm.input.sit import sit_reader
from libcbm.input.sit import sit_age_class_parser
from libcbm.model.cbm import cbm_simulator


class SITIntegrationTest(unittest.TestCase):

    def test_integration(self):

        sit = SimpleNamespace()
        sit.config = {
            "mapping_config": {
                "nonforest": None,
                "species": {
                    "species_classifier": "classifier1",
                    "species_mapping": [
                        {
                            "user_species": "a",
                            "default_species": "Spruce"
                        }
                    ]
                },
                "spatial_units": {
                    "mapping_mode": "SingleDefaultSpatialUnit",
                    "default_spuid": 42
                },
                "disturbance_types": [
                    {
                        "user_dist_type": "fire",
                        "default_dist_type": "Wildfire"
                    }
                ]
            }
        }

        sit.sit_data = sit_reader.parse(
            sit_classifiers=pd.DataFrame(
                data=[
                    (1, "_CLASSIFIER", "classifier1"),
                    (1, "a", "a")]),
            sit_disturbance_types=pd.DataFrame(data=[
                ("1", "fire")]),
            sit_age_classes=sit_age_class_parser.generate_sit_age_classes(
                5, 100),
            sit_inventory=pd.DataFrame(
                data=[("a", False, 100, 1, 0, 0, "1", "1")]),
            sit_yield=pd.DataFrame([
                ["a", "a"] +
                [x*15 for x in range(0, 20+1)]]),
            sit_events=None,
            sit_transitions=None
        )
        sit = sit_cbm_factory.initialize_sit_objects(sit)
        classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
        cbm = sit_cbm_factory.initialize_cbm(sit)
        results, reporting_func = \
            cbm_simulator.create_in_memory_reporting_func()
        cbm_simulator.simulate(
            cbm,
            n_steps=1,
            classifiers=classifiers,
            inventory=inventory,
            pool_codes=sit.defaults.get_pools(),
            flux_indicator_codes=sit.defaults.get_flux_indicators(),
            pre_dynamics_func=lambda time_step, cbm_vars: cbm_vars,
            reporting_func=reporting_func)
        # there should be 2 rows, timestep 0 and timestep 1
        self.assertTrue(results.pools.shape[0] == 2)
