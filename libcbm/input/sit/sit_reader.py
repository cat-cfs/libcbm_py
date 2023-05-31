# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Built-in modules #
from __future__ import annotations
import os
from typing import Iterable
from typing import Union

# Third party modules #
import pandas as pd

# Internal modules #
from libcbm.input.sit import sit_classifier_parser
from libcbm.input.sit import sit_disturbance_type_parser
from libcbm.input.sit import sit_age_class_parser
from libcbm.input.sit import sit_inventory_parser
from libcbm.input.sit import sit_yield_parser
from libcbm.input.sit import sit_disturbance_event_parser
from libcbm.input.sit import sit_transition_rule_parser
from libcbm.input.sit import sit_eligbility_parser


class SITData:
    def __init__(
        self,
        classifiers: pd.DataFrame,
        original_classifier_labels: list[str],
        classifier_values: pd.DataFrame,
        classifier_aggregates: pd.DataFrame,
        disturbance_types: pd.DataFrame,
        age_classes: pd.DataFrame,
        inventory: Union[pd.DataFrame, Iterable[pd.DataFrame]],
        yield_table: pd.DataFrame,
        disturbance_events: pd.DataFrame,
        transition_rules: pd.DataFrame,
        disturbance_eligibilities: pd.DataFrame = None,
        chunked_inventory: bool = False,
    ):
        self.classifiers = classifiers
        self.original_classifier_labels = original_classifier_labels
        self.classifier_values = classifier_values
        self.classifier_aggregates = classifier_aggregates
        self.disturbance_types = disturbance_types
        self.age_classes = age_classes
        self.inventory = inventory
        self.yield_table = yield_table
        self.disturbance_events = disturbance_events
        self.disturbance_eligibilities = disturbance_eligibilities
        self.transition_rules = transition_rules
        self.chunked_inventory = chunked_inventory


def load_table(config: dict, config_dir: str) -> pd.DataFrame:
    """Load a table based on the specified configuration.  The config_dir
    is used to compute absolute paths for file based tables.

    Supports The following formats:

        Excel or CSV:

        Uses pandas.read_csv or pandas.read_excel to load and return a
        pandas.DataFrame. With the exception of the "path" parameter, all
        parameters are passed as literal keyword args to the pandas.read_csv,
        or pandas.read_excel function.

        Examples::

            {"type": "csv"
             "params": {"path": "my_file.csv", "sep": "\\t"}

            {"type": "excel"
             "params: {"path": "my_file.xls", "header": null}

    Args:
        config (dict): configuration specifying a source of data
        config_dir (str): directory containing the configuration

    Raises:
        NotImplementedError: the name specified for "type" was not a
            supported data source.

    Returns:
        pandas.DataFrame: the loaded data
    """
    load_type = config["type"]
    load_params = config["params"]
    cwd = os.getcwd()
    try:
        if config_dir:
            os.chdir(config_dir)
        if load_type == "csv":
            path = os.path.abspath(os.path.relpath(load_params["path"]))
            load_params = load_params.copy()
            del load_params["path"]
            return pd.read_csv(path, **load_params)
        elif load_type == "excel":
            path = os.path.abspath(os.path.relpath(load_params["path"]))
            load_params = load_params.copy()
            del load_params["path"]
            return pd.read_excel(path, **load_params)
        else:
            raise NotImplementedError(
                f"The specified table type {load_type} is not supported."
            )
    finally:
        os.chdir(cwd)


def read(config: dict, config_dir: str) -> SITData:
    # Call pandas.read_csv on all input files #
    sit_classifiers = load_table(config["classifiers"], config_dir)
    sit_disturbance_types = load_table(config["disturbance_types"], config_dir)
    sit_age_classes = load_table(config["age_classes"], config_dir)
    sit_inventory = load_table(config["inventory"], config_dir)
    sit_yield = load_table(config["yield"], config_dir)
    sit_events = (
        load_table(config["events"], config_dir)
        if "events" in config and config["events"]
        else None
    )
    sit_eligibilities = (
        load_table(config["eligibilities"], config_dir)
        if "eligibilities" in config and config["eligibilities"]
        else None
    )
    sit_transitions = (
        load_table(config["transitions"], config_dir)
        if "transitions" in config and config["transitions"]
        else None
    )
    parse_options = None
    if "parse_options" in config:
        parse_options = SITParseOptions(
            inventory_ids=config["parse_options"]["sit_inventory_ids"],
            event_ids=config["parse_options"]["sit_event_ids"],
            events_external_eligibilities=config["parse_options"][
                "sit_events_external_eligibilities"
            ],
            transitions_external_eligibilities=config["parse_options"][
                "sit_transitions_external_eligibilities"
            ],
        )
    # Validate data #
    sit_data = parse(
        sit_classifiers,
        sit_disturbance_types,
        sit_age_classes,
        sit_inventory,
        sit_yield,
        sit_events,
        sit_transitions,
        sit_eligibilities,
        parse_options,
    )
    # Return #
    return sit_data


class SITParseOptions:
    def __init__(
        self,
        inventory_ids: bool = False,
        event_ids: bool = False,
        events_external_eligibilities: bool = False,
        transitions_external_eligibilities: bool = False,
    ):
        self._inventory_ids = inventory_ids
        self._event_ids = event_ids
        self._events_external_eligibilities = events_external_eligibilities
        self._transitions_external_eligibilities = (
            transitions_external_eligibilities
        )

    @property
    def inventory_ids(self) -> bool:
        return self._inventory_ids

    @property
    def event_ids(self) -> bool:
        return self._event_ids

    @property
    def events_external_eligibilities(self) -> bool:
        return self._events_external_eligibilities

    @property
    def transitions_external_eligibilities(self) -> bool:
        return self._transitions_external_eligibilities


def parse(
    sit_classifiers: pd.DataFrame,
    sit_disturbance_types: pd.DataFrame,
    sit_age_classes: pd.DataFrame,
    sit_inventory: Union[pd.DataFrame, Iterable[pd.DataFrame]],
    sit_yield: pd.DataFrame,
    sit_events: pd.DataFrame = None,
    sit_transitions: pd.DataFrame = None,
    sit_eligibilities: pd.DataFrame = None,
    sit_parse_options: SITParseOptions = None,
) -> SITData:
    """Parses and validates CBM Standard import tool formatted data including
    the complicated interdependencies in the SIT format. Returns an object
    containing the validated result.

    The returned object has the following properties:

     - classifiers: a pandas.DataFrame of classifiers in the sit_classifiers
        input
     - classifier_values: a pandas.DataFrame of the classifier values in the
        sit_classifiers input
     - classifier_aggregates: a dictionary of the classifier aggregates
        in the sit_classifiers input
     - disturbance_types: a pandas.DataFrame based on the disturbance types in
        the sit_disturbance_types input
     - age_classes: a pandas.DataFrame of the age classes based on
        sit_age_classes
     - inventory: a pandas.DataFrame of the inventory based on sit_inventory
     - yield_table: a pandas.DataFrame of the merchantable volume yield curves
        in the sit_yield input
     - disturbance_events: a pandas.DataFrame of the disturbance events based
        on sit_events.  If the sit_events parameter is None this field is None.
     - transition_rules: a pandas.DataFrame of the transition rules based on
        sit_transitions.  If the sit_transitions parameter is None this field
        is None.
     - disturbance_eligibilities: a pandas.DataFrame of the disturbance event
        eligibilities based on sit_eligibilities.  If the sit_events parameter
        is None this field is None.

    Args:
        sit_classifiers (pandas.DataFrame): SIT formatted classifiers
        sit_disturbance_types (pandas.DataFrame): SIT formatted disturbance
            types
        sit_age_classes (pandas.DataFrame): SIT formatted age classes
        sit_inventory (pandas.DataFrame): SIT formatted inventory
        sit_yield (pandas.DataFrame): SIT formatted yield curves
        sit_events (pandas.DataFrame, optional): SIT formatted disturbance
            events
        sit_transitions (pandas.DataFrame, optional): SIT formatted transition
            rules. Defaults to None.
        sit_eligibilities (pandas.DataFrame, optional): SIT formatted
            disturbance eligibilities. Defaults to None.

    Returns:
        SITData: an object containing parsed and validated SIT dataset
    """

    if not sit_parse_options:
        sit_parse_options = SITParseOptions()

    (
        classifiers,
        original_classifier_labels,
        classifier_values,
        classifier_aggregates,
    ) = sit_classifier_parser.parse(sit_classifiers)

    disturbance_types = sit_disturbance_type_parser.parse(
        sit_disturbance_types
    )
    age_classes = sit_age_class_parser.parse(sit_age_classes)

    if isinstance(sit_inventory, pd.DataFrame):
        is_chunked_inventory = False

        inventory = sit_inventory_parser.parse(
            sit_inventory,
            classifiers,
            classifier_values,
            disturbance_types,
            age_classes,
            sit_parse_options.inventory_ids,
        )
    else:
        raise NotImplementedError("")

    yield_table = sit_yield_parser.parse(
        sit_yield, classifiers, classifier_values, age_classes
    )

    if sit_events is not None:
        disturbance_events = sit_disturbance_event_parser.parse(
            sit_events,
            classifiers,
            classifier_values,
            classifier_aggregates,
            disturbance_types,
            age_classes,
            sit_parse_options.events_external_eligibilities,
            sit_parse_options.event_ids,
        )
    else:
        disturbance_events = None

    if sit_transitions is not None:
        transition_rules = sit_transition_rule_parser.parse(
            sit_transitions,
            classifiers,
            classifier_values,
            classifier_aggregates,
            disturbance_types,
            age_classes,
            sit_parse_options.transitions_external_eligibilities,
        )
    else:
        transition_rules = None

    if (
        sit_parse_options.events_external_eligibilities
        or sit_parse_options.transitions_external_eligibilities
    ):
        # if either of the above are true, require separate eligbilites file
        if sit_eligibilities is None:
            raise ValueError(
                "sit_eligibilites must be specified with "
                "sit_events_external_eligibilities or "
                "sit_transitions_external_eligibilities options enabled"
            )
        eligibilities = sit_eligbility_parser.parse_eligibilities(
            sit_eligibilities
        )
        sit_eligbility_parser.validate_eligibilities_relationship(
            eligibilities,
            disturbance_events=(
                disturbance_events
                if sit_parse_options.events_external_eligibilities
                else None
            ),
            transition_rules=(
                transition_rules
                if sit_parse_options._transitions_external_eligibilities
                else None
            ),
        )
    else:
        eligibilities = None

    return SITData(
        classifiers,
        original_classifier_labels,
        classifier_values,
        classifier_aggregates,
        disturbance_types,
        age_classes,
        inventory,
        yield_table,
        disturbance_events,
        transition_rules,
        eligibilities,
        is_chunked_inventory,
    )
