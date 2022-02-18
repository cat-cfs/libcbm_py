import os
import json
from contextlib import contextmanager
from types import SimpleNamespace
import pandas as pd

from libcbm.input.sit import sit_cbm_factory
from libcbm.input.sit import sit_classifier_parser
from libcbm.input.sit import sit_transition_rule_parser
from libcbm.input.sit import sit_format
from libcbm.model.cbm import cbm_defaults
from libcbm import resources
from libcbm.input.sit import sit_reader
from libcbm.model.cbm import cbm_variables
from libcbm.model.cbm.rule_based.classifier_filter import ClassifierFilter
from libcbm.model.cbm.rule_based.sit import sit_transition_rule_processor
from libcbm.model.cbm.rule_based.transition_rule_processor import (
    TransitionRuleProcessor,
)


def get_parameters_factory(sit):
    """overrides selected default parameters for testing purposes.

    For example since rule based disturbances use the matrix flows into the
    "Products" pool, set up disturbance matrices so that this is predictable.

    Returns:
        func: a function to be passed to initialize CBM default parameters
    """

    parameters = sit.get_parameters_factory()()
    disturbance_matrix_value = parameters["disturbance_matrix_values"]

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
        pool["id"]
        for pool in pools
        if pool["name"]
        not in [
            "SoftwoodMerch",
            "HardwoodMerch",
            "SoftwoodStemSnag",
            "HardwoodStemSnag",
            "Input",
            "CO2",
            "CO",
            "CH4",
            "N2O",
            "Products",
        ]
    ]
    # set up a matrix where all merch and snags flow into the products pool
    matrix_sources = [
        sw_merch,
        hw_merch,
        sw_stem_snag,
        hw_stem_snag,
    ] + non_source_pools
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
                    "proportion": matrix_values,
                },
                columns=[
                    "disturbance_matrix_id",
                    "source_pool_id",
                    "sink_pool_id",
                    "proportion",
                ],
            )
        )
    new_matrix = new_matrix.reset_index(drop=True)

    parameters["disturbance_matrix_values"] = new_matrix

    def parameters_factory():
        return parameters

    return parameters_factory


def get_test_data_dir():
    return os.path.join(
        resources.get_test_resources_dir(), "sit_rule_based_events"
    )


def load_sit_input():
    sit_input = SimpleNamespace()
    sit_config = load_config()
    sit_input.sit_data = sit_reader.read(
        sit_config["import_config"], get_test_data_dir()
    )
    sit_input.config = sit_config
    return sit_input


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


def initialize_transitions(sit, transition_data):
    return df_from_template_row(
        template_row=sit.sit_data.transition_rules.iloc[0],
        row_dicts=transition_data,
    )


def initialize_events(sit_input, event_data):
    # the first row is a template row, and the specified dict will replace the
    # values
    return df_from_template_row(
        template_row=sit_input.sit_data.disturbance_events.iloc[0],
        row_dicts=event_data,
    )


def initialize_inventory(sit_input, inventory_data):
    return df_from_template_row(
        template_row=sit_input.sit_data.inventory.iloc[0], row_dicts=inventory_data
    )


def setup_cbm_vars(sit):

    classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)

    cbm_vars = cbm_variables.initialize_simulation_variables(
        classifiers,
        inventory,
        sit.defaults.get_pools(),
        sit.defaults.get_flux_indicators(),
    )
    return cbm_vars


@contextmanager
def get_rule_based_processor(sit, random_func=None, parameters_factory=None):

    with sit_cbm_factory.initialize_cbm(
        sit, dll_path=None, parameters_factory=parameters_factory
    ) as cbm:
        rule_based_processor = sit_cbm_factory.create_sit_rule_based_processor(
            sit, cbm, random_func
        )
        yield rule_based_processor


def get_disturbance_type_ids(sit_disturbance_types, disturbance_types):
    result = []
    sit_disturbance_type_id_lookup = {
        x["id"]: x["sit_disturbance_type_id"]
        for _, x in sit_disturbance_types.iterrows()
    }

    for d in disturbance_types:
        if d in sit_disturbance_type_id_lookup:
            result.append(sit_disturbance_type_id_lookup[d])
        else:
            result.append(0)
    return result


def get_transition_rules_pre_dynamics_func(sit):

    classifiers_config = sit_cbm_factory.get_classifiers(
        sit.sit_data.classifiers, sit.sit_data.classifier_values
    )
    sit_transition_rules = sit_cbm_factory.initialize_transition_rules(sit)
    classifier_filter = ClassifierFilter(
        classifiers_config=classifiers_config,
        classifier_aggregates=sit.sit_data.classifier_aggregates,
    )
    classifier_value_post_fix = sit_format.get_tr_classifier_set_postfix()
    group_err_max = sit_transition_rule_parser.GROUPED_PERCENT_ERR_MAX
    state_filter_func = (
        sit_transition_rule_processor.state_variable_filter_func
    )
    tr_processor = TransitionRuleProcessor(
        classifier_filter_builder=classifier_filter,
        state_variable_filter_func=state_filter_func,
        classifiers_config=classifiers_config,
        grouped_percent_err_max=group_err_max,
        wildcard=sit_classifier_parser.get_wildcard_keyword(),
        transition_classifier_postfix=classifier_value_post_fix,
    )
    sit_tr_processor = (
        sit_transition_rule_processor.SITTransitionRuleProcessor(tr_processor)
    )
    return sit_transition_rule_processor.get_pre_dynamics_func(
        sit_tr_processor, sit_transition_rules
    )
