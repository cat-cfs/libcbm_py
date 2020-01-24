import unittest
import os

from libcbm.input.sit import sit_cbm_factory
from libcbm.model.cbm import cbm_simulator
from libcbm import resources


class SITCBMFactoryTest(unittest.TestCase):

    def test_integration_with_tutorial2(self):
        """tests full CBM integration with rule based disturbances
        """
        config_path = os.path.join(
            resources.get_test_resources_dir(),
            "cbm3_tutorial2", "sit_config.json")
        sit = sit_cbm_factory.load_sit(config_path)
        classifiers, inventory = \
            sit_cbm_factory.initialize_inventory(sit)
        cbm = sit_cbm_factory.initialize_cbm(sit)
        results, reporting_func = \
            cbm_simulator.create_in_memory_reporting_func()
        rule_based_stats = sit_cbm_factory.create_rule_based_stats()
        rule_based_event_func = \
            sit_cbm_factory.create_sit_rule_based_pre_dynamics_func(
                sit, cbm, rule_based_stats.append_stats)
        cbm_simulator.simulate(
            cbm,
            n_steps=1,
            classifiers=classifiers,
            inventory=inventory,
            pool_codes=sit.defaults.get_pools(),
            flux_indicator_codes=sit.defaults.get_flux_indicators(),
            pre_dynamics_func=rule_based_event_func,
            reporting_func=reporting_func)
        self.assertTrue(
            results.pools[results.pools.timestep == 0].shape[0]
            == inventory.shape[0])
        self.assertTrue(rule_based_stats.stats.shape[0] > 0)
