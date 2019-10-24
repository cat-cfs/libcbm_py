import unittest
import os
import json
from types import SimpleNamespace
import pandas as pd
import numpy as np
from mock import Mock
from libcbm.input.sit import sit_cbm_factory
from libcbm.input.sit import sit_reader
from libcbm.model.cbm import cbm_variables

from libcbm.model.cbm.rule_based.classifier_filter import ClassifierFilter
from libcbm.model.cbm.rule_based.sit import sit_event_processor
from libcbm.model.cbm import cbm_defaults
from libcbm import resources

FIRE_ID = 1
CLEARCUT_ID = 3
DEFORESTATION_ID = 7

def get_parameters_factory():
    """overrides selected default parameters for testing purposes.

    For example since rule based disturbances use the matrix flows into the
    "Products" pool, set up disturbance matrices so that this is predictable.

    Returns:
        func: a function to be passed to initialize CBM default parameters
    """
    parameters = cbm_defaults.load_cbm_parameters(
        resources.get_cbm_defaults_path())
    disturbance_matrix_value = cbm_defaults.parameter_as_dataframe(
        parameters["disturbance_matrix_values"])

    # since for the purposes of rule based disturbances we are
    # only really interested in production results of disturbance matrices
    # set some easy to work with DM values.
    dmids = list(disturbance_matrix_value.disturbance_matrix_id.unique())

    pools = cbm_defaults.load_cbm_pools(resources.get_cbm_defaults_path())
    pool_ids = {pool["name"]: pool["id"] for pool in pools}
    sw_merch = pool_ids["SoftwoodMerch"]
    hw_merch = pool_ids["HardwoodMerch"]
    sw_stem_snag = pool_ids["SoftwoodStemSnag"]
    hw_stem_snag = pool_ids["HardwoodStemSnag"]
    products = pool_ids["Products"]
    non_source_pools = [
        pool["id"] for pool in pools if pool["name"] not in
        ["SoftwoodMerch", "HardwoodMerch", "SoftwoodStemSnag",
         "HardwoodStemSnag", "Input", "CO2", "CO", "CH4", "N2O",
         "Products"]]
    # set up a matrix where all merch and snags flow into the products pool
    matrix_sources = [
        sw_merch, hw_merch, sw_stem_snag, hw_stem_snag] + non_source_pools
    matrix_sinks = [products] * 4 + non_source_pools
    matrix_values = [1.0] * len(matrix_sources)
    new_matrix = pd.DataFrame()
    for dmid in dmids:
        new_matrix = new_matrix.append(
            pd.DataFrame(
                data={
                    "disturbance_matrix_id": dmid,
                    "source_pool_id": matrix_sources,
                    "sink_pool_id": matrix_sinks,
                    "proportion": matrix_values},
                columns=[
                    "disturbance_matrix_id", "source_pool_id", "sink_pool_id",
                    "proportion"]))
    new_matrix = new_matrix.reset_index(drop=True)

    parameters["disturbance_matrix_values"] = \
        cbm_defaults.dataframe_as_parameter(new_matrix)

    def parameters_factory():
        return parameters
    return parameters_factory


def get_test_data_dir():
    return os.path.join(
        resources.get_test_resources_dir(), "sit_rule_based_events")


def load_sit_data():
    sit = SimpleNamespace()
    sit_config = load_config()
    sit.sit_data = sit_reader.read(
        sit_config["import_config"], get_test_data_dir())
    sit.config = sit_config
    return sit


def load_config():
    sit_rule_based_examples_dir = get_test_data_dir()

    config_path = os.path.join(sit_rule_based_examples_dir, "sit_config.json")
    with open(config_path) as sit_config_fp:
        sit_config = json.load(sit_config_fp)
    return sit_config


def df_from_template_row(template_row, row_dicts):
    result = pd.DataFrame()
    for data in row_dicts:
        new_row = template_row.copy()
        for key, value in data.items():
            new_row.loc[key] = value

        result = result.append(new_row)
    return result.reset_index(drop=True)


def initialize_events(sit, event_data):
    # the first row is a template row, and the specified dict will replace the
    # values
    return df_from_template_row(
        template_row=sit.sit_data.disturbance_events.iloc[0],
        row_dicts=event_data)


def initialize_inventory(sit, inventory_data):
    return df_from_template_row(
        template_row=sit.sit_data.inventory.iloc[0],
        row_dicts=inventory_data)


def setup_cbm_vars(sit):

    sit = sit_cbm_factory.initialize_sit_objects(sit)

    classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)

    cbm_vars = cbm_variables.initialize_simulation_variables(
        classifiers, inventory, sit.defaults.get_pools(),
        sit.defaults.get_flux_indicators())
    return cbm_vars


def get_pre_dynamics_func(sit, on_unrealized, parameters_factory=None,
                          random_func=None):

    sit_events = sit_cbm_factory.initialize_events(sit)
    cbm = sit_cbm_factory.initialize_cbm(
        sit, parameters_factory=parameters_factory)
    classifier_filter = ClassifierFilter(
        classifiers_config=sit_cbm_factory.get_classifiers(
            sit.sit_data.classifiers, sit.sit_data.classifier_values),
        classifier_aggregates=sit.sit_data.classifier_aggregates)
    processor = sit_event_processor.SITEventProcessor(
        model_functions=cbm.model_functions,
        compute_functions=cbm.compute_functions,
        cbm_defaults_ref=sit.defaults,
        classifier_filter_builder=classifier_filter,
        random_generator=random_func,
        on_unrealized_event=on_unrealized)
    return sit_event_processor.get_pre_dynamics_func(
        processor, sit_events)


class SITEventIntegrationTest(unittest.TestCase):

    def test_rule_based_area_target_age_sort(self):
        """Test a rule based event with area target, and age sort where no
        splitting occurs
        """
        mock_on_unrealized = Mock()
        sit = load_sit_data()
        sit.sit_data.disturbance_events = initialize_events(sit, [
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_SW_AGE", "target_type": "Area",
             "target": 10, "disturbance_type": "fire", "time_step": 1}
        ])

        # records 0, 2, and 3 match, and 1 does not.  The target is 10, so
        # 2 of the 3 eligible records will be disturbed
        sit.sit_data.inventory = initialize_inventory(sit, [
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a2", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5}
        ])

        cbm_vars = setup_cbm_vars(sit)

        # since age sort is set, the oldest values of the eligible records
        # will be disturbed
        cbm_vars.state.age = np.array([99, 100, 98, 100])

        pre_dynamics_func = get_pre_dynamics_func(sit, mock_on_unrealized)
        cbm_vars_result = pre_dynamics_func(time_step=1, cbm_vars=cbm_vars)

        # records 0 and 3 are the disturbed records: both are eligible, they
        # are the oldest stands, and together they exactly satisfy the target.
        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) ==
                [FIRE_ID, 0, 0, FIRE_ID])

    def test_rule_based_area_target_age_sort_unrealized(self):
        """Test a rule based event with area target, and age sort where no
        splitting occurs
        """
        mock_on_unrealized = Mock()

        sit = load_sit_data()
        sit.sit_data.disturbance_events = initialize_events(sit, [
            {"admin": "a2", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_SW_AGE", "target_type": "Area",
             "target": 10, "disturbance_type": "fire", "time_step": 1}
        ])

        # record at index 1 is the only eligible record meaning the above event
        # will be unrealized with a shortfall of 5
        sit.sit_data.inventory = initialize_inventory(sit, [
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a2", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5}
        ])

        cbm_vars = setup_cbm_vars(sit)

        # since age sort is set, the oldest values of the eligible records
        # will be disturbed
        cbm_vars.state.age = np.array([99, 100, 98, 100])

        pre_dynamics_func = get_pre_dynamics_func(sit, mock_on_unrealized)
        cbm_vars_result = pre_dynamics_func(time_step=1, cbm_vars=cbm_vars)

        # records 0 and 3 are the disturbed records: both are eligible, they
        # are the oldest stands, and together they exactly satisfy the target.
        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) == [0, 1, 0, 0])

        mock_args, _ = mock_on_unrealized.call_args
        self.assertTrue(mock_args[0] == 5)
        expected = sit.sit_data.disturbance_events.to_dict("records")[0]
        expected["disturbance_type_id"] = 1
        diff = set(mock_args[1].items()) ^ set(expected.items())
        self.assertTrue(len(diff) == 0)

    def test_rule_based_area_target_age_sort_multiple_event(self):
        """Check interactions between two age sort/area target events
        """
        mock_on_unrealized = Mock()
        sit = load_sit_data()
        sit.sit_data.disturbance_events = initialize_events(sit, [
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_SW_AGE", "target_type": "Area",
             "target": 10, "disturbance_type": "clearcut",
             "time_step": 1},
            {"admin": "?", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_SW_AGE", "target_type": "Area",
             "target": 10, "disturbance_type": "fire", "time_step": 1},
        ])
        # the second of the above events will match all records, and it will
        # occur first since fire happens before clearcut

        sit.sit_data.inventory = initialize_inventory(sit, [
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a2", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e3", "species": "sp", "area": 5}
        ])

        cbm_vars = setup_cbm_vars(sit)

        # since age sort is set, the oldest values of the eligible records
        # will be disturbed
        cbm_vars.state.age = np.array([100, 99, 98, 97, 96])

        pre_dynamics_func = get_pre_dynamics_func(sit, mock_on_unrealized)
        cbm_vars_result = pre_dynamics_func(time_step=1, cbm_vars=cbm_vars)

        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) ==
            [FIRE_ID, FIRE_ID, 0, CLEARCUT_ID, CLEARCUT_ID])

    def test_rule_based_area_target_age_sort_split(self):
        """Test a rule based event with area target, and age sort where no
        splitting occurs
        """
        mock_on_unrealized = Mock()
        sit = load_sit_data()
        sit.sit_data.disturbance_events = initialize_events(sit, [
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_SW_AGE", "target_type": "Area",
             "target": 6, "disturbance_type": "fire", "time_step": 1}
        ])
        # since the target is 6, one of the 2 inventory records below needs to
        # be split
        sit.sit_data.inventory = initialize_inventory(sit, [
            {"admin": "a1", "eco": "e1", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5}
        ])

        cbm_vars = setup_cbm_vars(sit)

        # since the sort is by age, the first record will be fully disturbed
        # and the second will be split into 1 and 4 hectare stands.
        cbm_vars.state.age = np.array([99, 100])

        pre_dynamics_func = get_pre_dynamics_func(sit, mock_on_unrealized)
        cbm_vars_result = pre_dynamics_func(time_step=1, cbm_vars=cbm_vars)

        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) ==
            [FIRE_ID, FIRE_ID, 0])

        self.assertTrue(cbm_vars.pools.shape[0] == 3)
        self.assertTrue(cbm_vars.flux_indicators.shape[0] == 3)
        self.assertTrue(cbm_vars.state.shape[0] == 3)
        # note the age sort order caused the first record to split
        self.assertTrue(list(cbm_vars.inventory.area) == [1, 5, 4])

    def test_rule_based_merch_target_age_sort(self):
        mock_on_unrealized = Mock()
        sit = load_sit_data()

        sit.sit_data.disturbance_events = initialize_events(sit, [
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_HW_AGE", "target_type": "Merchantable",
             "target": 10, "disturbance_type": "clearcut",
             "time_step": 1}
        ])

        sit.sit_data.inventory = initialize_inventory(sit, [
            {"admin": "a1", "eco": "e1", "species": "sp", "area": 5},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5}
        ])

        cbm_vars = setup_cbm_vars(sit)

        # 1 tonnes C/ha * 10 ha total = 10 tonnes C
        cbm_vars.pools.SoftwoodMerch = 1.0
        cbm_vars.state.age = np.array([99, 100])

        pre_dynamics_func = get_pre_dynamics_func(
            sit, mock_on_unrealized, get_parameters_factory())
        cbm_vars_result = pre_dynamics_func(time_step=1, cbm_vars=cbm_vars)

        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) ==
            [CLEARCUT_ID, CLEARCUT_ID])

    def test_rule_based_merch_target_age_sort_unrealized(self):
        mock_on_unrealized = Mock()
        sit = load_sit_data()

        sit.sit_data.disturbance_events = initialize_events(sit, [
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_HW_AGE", "target_type": "Merchantable",
             "target": 10, "disturbance_type": "clearcut",
             "time_step": 1}
        ])

        sit.sit_data.inventory = initialize_inventory(sit, [
            {"admin": "a1", "eco": "e1", "species": "sp", "area": 3},
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 4},
            {"admin": "a1", "eco": "e1", "species": "sp", "area": 2},
        ])

        cbm_vars = setup_cbm_vars(sit)

        # 1 tonnes C/ha * (3+4+2) ha total = 9 tonnes C available for event,
        # with target = 10, therefore the expected shortfall is 1
        cbm_vars.pools.SoftwoodMerch = 1.0
        cbm_vars.state.age = np.array([99, 100, 98])

        pre_dynamics_func = get_pre_dynamics_func(
            sit, mock_on_unrealized, get_parameters_factory())
        cbm_vars_result = pre_dynamics_func(time_step=1, cbm_vars=cbm_vars)

        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) ==
            [CLEARCUT_ID, CLEARCUT_ID, CLEARCUT_ID])

        mock_args, _ = mock_on_unrealized.call_args
        # confirm expected shortfall (in tonnes C)
        self.assertTrue(mock_args[0] == 1)
        expected = sit.sit_data.disturbance_events.to_dict("records")[0]
        expected["disturbance_type_id"] = 3
        diff = set(mock_args[1].items()) ^ set(expected.items())
        self.assertTrue(len(diff) == 0)

    def test_rule_based_merch_target_age_sort_split(self):
        mock_on_unrealized = Mock()
        sit = load_sit_data()

        sit.sit_data.disturbance_events = initialize_events(sit, [
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_HW_AGE", "target_type": "Merchantable",
             "target": 7, "disturbance_type": "clearcut",
             "time_step": 4}
        ])

        sit.sit_data.inventory = initialize_inventory(sit, [
            # the remaining target after 7 - 5 = 2, so 2/5ths of the area
            # of this stand will be disturbed
            {"admin": "a1", "eco": "e1", "species": "sp", "area": 5},
            # this entire record will be disturbed first (see age sort)
            {"admin": "a1", "eco": "e2", "species": "sp", "area": 5}
        ])

        cbm_vars = setup_cbm_vars(sit)

        cbm_vars.pools.SoftwoodMerch = 1.0
        cbm_vars.state.age = np.array([99, 100])

        pre_dynamics_func = get_pre_dynamics_func(
            sit, mock_on_unrealized, get_parameters_factory())
        cbm_vars_result = pre_dynamics_func(time_step=4, cbm_vars=cbm_vars)

        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) ==
            [CLEARCUT_ID, CLEARCUT_ID, 0])
        self.assertTrue(cbm_vars.pools.shape[0] == 3)
        self.assertTrue(cbm_vars.flux_indicators.shape[0] == 3)
        self.assertTrue(cbm_vars.state.shape[0] == 3)
        # note the age sort order caused the first record to split
        self.assertTrue(list(cbm_vars.inventory.area) == [2, 5, 3])

    def test_rule_based_multiple_target_types(self):
        mock_on_unrealized = Mock()
        sit = load_sit_data()

        sit.sit_data.disturbance_events = initialize_events(sit, [
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "MERCHCSORT_TOTAL", "target_type": "Merchantable",
             "target": 100, "disturbance_type": "clearcut",
             "time_step": 100},
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "SORT_BY_SW_AGE", "target_type": "Area",
             "target": 20, "disturbance_type": "deforestation",
             "time_step": 100},
            # this event will occur first
            {"admin": "a1", "eco": "?", "species": "sp",
             "sort_type": "RANDOMSORT", "target_type": "Area",
             "target": 20, "disturbance_type": "fire",
             "time_step": 100},
        ])

        sit.sit_data.inventory = initialize_inventory(sit, [
            {"admin": "a1", "eco": "e1", "species": "sp", "area": 1000},
        ])

        cbm_vars = setup_cbm_vars(sit)

        cbm_vars.pools.HardwoodMerch = 1.0
        cbm_vars.state.age = np.array([50])

        pre_dynamics_func = get_pre_dynamics_func(
            sit, mock_on_unrealized, get_parameters_factory(),
            random_func=np.ones)
        cbm_vars_result = pre_dynamics_func(time_step=100, cbm_vars=cbm_vars)
        self.assertTrue(
            list(cbm_vars_result.params.disturbance_type) ==
            [FIRE_ID, CLEARCUT_ID, DEFORESTATION_ID, 0])
        self.assertTrue(list(cbm_vars.inventory.area) == [20, 100, 20, 860])
