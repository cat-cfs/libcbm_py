import pandas as pd
import numpy as np

from libcbm import resources
from libcbm.model.cbm import cbm_config
from libcbm.model.cbm import cbm_defaults
from libcbm.model.cbm import cbm_factory
from libcbm.model.cbm import cbm_variables
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference
from libcbm.model.cbm import cbm_simulator


def classifiers_factory():
    return cbm_config.classifier_config(
        [cbm_config.classifier(
            "c1", values=[cbm_config.classifier_value("c1_v1")])])


def create_merch_volumes_factory(db_path):
    def factory():
        return cbm_config.merch_volume_to_biomass_config(
            db_path=db_path,
            merch_volume_curves=[
                cbm_config.merch_volume_curve(
                    classifier_set=["c1_v1"],
                    merch_volumes=[
                        {
                            "species_id": 1,
                            "age_volume_pairs": [
                                [0, 0], [50, 100], [100, 150], [150, 200]],
                        }
                    ],
                )
            ],
        )
    return factory


def test_integration():
    db_path = resources.get_cbm_defaults_path()

    with cbm_factory.create(
        dll_path=resources.get_libcbm_bin_path(),
        dll_config_factory=cbm_defaults.get_libcbm_configuration_factory(
            db_path),
        cbm_parameters_factory=cbm_defaults.get_cbm_parameters_factory(
            db_path),
        merch_volume_to_biomass_factory=create_merch_volumes_factory(db_path),
        classifiers_factory=classifiers_factory
    ) as cbm:

        n_stands = 2
        n_steps = 10
        ref = CBMDefaultsReference(db_path)

        inventory = cbm_variables.initialize_inventory(
            inventory=pd.DataFrame({
                "age": np.full(n_stands, 15, dtype=int) * 15,
                "area": np.ones(n_stands),
                "spatial_unit": np.ones(n_stands, dtype=int) * 16,
                "afforestation_pre_type_id": np.full(n_stands, -1, dtype=int),
                "land_class": np.zeros(n_stands, dtype=int),
                "historical_disturbance_type": np.full(n_stands, 1, dtype=int),
                "last_pass_disturbance_type": np.full(n_stands, 1, dtype=int),
                "delay": np.zeros(n_stands, dtype=int),
            }))
        classifiers = pd.DataFrame({
                "c1": np.ones(n_stands, dtype=int) * 1
            })

        spinup_results, spinup_reporting_func = \
            cbm_simulator.create_in_memory_reporting_func(density=True)
        cbm_results, cbm_reporting_func = \
            cbm_simulator.create_in_memory_reporting_func(
                classifier_map={1: "c1_v1"}, disturbance_type_map={1: "fires"})
        cbm_simulator.simulate(
            cbm,
            n_steps=n_steps,
            classifiers=classifiers,
            inventory=inventory,
            pool_codes=ref.get_pools(),
            flux_indicator_codes=ref.get_flux_indicators(),
            pre_dynamics_func=lambda t, cbm_vars: cbm_vars,
            reporting_func=cbm_reporting_func,
            spinup_params=cbm_variables.initialize_spinup_parameters(
                n_stands, 50, 5, 5, -1),
            spinup_reporting_func=spinup_reporting_func)
        assert len(cbm_results.pools.index) == (n_steps + 1) * n_stands
