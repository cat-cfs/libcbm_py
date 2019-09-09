import pandas as pd
import numpy as np
from libcbm.model.cbm import cbm_defaults
from libcbm.model.cbm import cbm_factory
from libcbm.model.cbm import cbm_config


def get_classifiers(classifiers, classifier_values):
    """Create classifier input for initializing the CBM class based on CBM
    Standard import tool formatted data.

    Args:
        classifiers (pandas.DataFrame): the parsed SIT classifiers output
            of :py:func:`libcbm.input.sit.sit_classifier_parser.parse`
        classifier_values (pandas.DataFrame): the parsed SIT classifier values
            output of :py:func:`libcbm.input.sit.sit_classifier_parser.parse`

    Returns:
        dict: configuration dictionary for CBM. See:
            :py:func:`libcbm.model.cbm.cbm_config.classifier_config`
    """
    classifiers_config = []
    for _, row in classifiers.iterrows():
        values = classifier_values[
            classifier_values.classifier_id == row.id].name
        classifiers_config.append(cbm_config.classifier(
            name=row["name"],
            values=[cbm_config.classifier_value(value=x) for x in list(values)]
        ))

    config = cbm_config.classifier_config(classifiers_config)
    return config


def get_merch_volumes(yield_table, classifiers, age_classes):
    """Create merchantable volume input for initializing the CBM class
    based on CBM Standard import tool formatted data.

    Args:
        yield_table (pandas.DataFrame): the parsed SIT yield output
            of :py:func:`libcbm.input.sit.sit_yield_parser.parse`
        classifiers (pandas.DataFrame): the parsed SIT classifiers output
            of :py:func:`libcbm.input.sit.sit_classifier_parser.parse`
        age_classes (pandas.DataFrame): the parsed SIT age classes
            output of :py:func:`libcbm.input.sit.sit_age_class_parser.parse`

    Returns:
        dict: configuration dictionary for CBM. See:
            :py:func:`libcbm.model.cbm.cbm_config.classifier_config`
    """

    unique_classifier_sets = yield_table.groupby(
        list(classifiers.name)).size().reset_index()
    # removes the extra field created by the above method
    unique_classifier_sets = unique_classifier_sets.drop(columns=[0])
    ages = list(age_classes.end_year)
    output = []
    for _, row in unique_classifier_sets.iterrows():
        match = yield_table.merge(
            pd.DataFrame([row]),
            left_on=list(classifiers.name),
            right_on=list(classifiers.name))
        merch_vols = []
        for _, match_row in match.iterrows():
            vols = match_row.iloc[len(classifiers)+1:]
            merch_vols.append({
                "species_id": match_row["leading_species"],
                "age_volume_pairs": [
                    (ages[i], vols[i]) for i in range(len(vols))]
            })
        output.append(
            cbm_config.merch_volume_curve(
                classifier_set=list(row),
                merch_volumes=merch_vols
            ))
    return output


def initialize_inventory(inventory, classifiers, classifier_values,
                         cbm_defaults_ref):

    classifier_config = get_classifiers(classifiers, classifier_values)
    classifier_ids = [
        (x["id"], x["name"]) for x in classifier_config["classifiers"]]
    classifier_value_id_lookups = {}

    for identifier, name in classifier_ids:
        classifier_value_id_lookups[name] = {
            x["value"]: x["id"]
            for x in classifier_config["classifier_values"]
            if x["classifier_id"] == identifier}
    classifiers_result = pd.DataFrame(
        data={
            name: inventory[name].map(classifier_value_id_lookups[name])
            for name in list(classifiers.name)},
        columns=list(classifiers.name))
    classifiers_result = np.ascontiguousarray(
        classifiers_result, dtype=np.int32)
    inventory_result = pd.DataFrame(
        data={
            "age": inventory.age,
            "spatial_unit": 42,  # TODO: use "classifier mapping" to determine spu
            "afforestation_pre_type_id": 0,  # TODO: use "classifier mapping" to determine non-forest cover
            "area": inventory.area,
            "delay": inventory.delay,
            "land_class": inventory.land_class.map(
                cbm_defaults_ref.get_land_class_id),
            "historical_disturbance_type": 1,  # TODO: use "disturbance type mapping" to determine historic and last pass
            "last_pass_disturbance_type": 1,
        })
    return classifiers_result, inventory_result


def initialize_cbm(db_path, dll_path, yield_table, classifiers,
                   classifier_values, age_classes):
    """Create an initialized instance of
        :py:class:`libcbm.model.cbm.cbm_model.CBM` based on SIT input

    Args:
        db_path (str): path to a cbm_defaults database
        dll_path (str): path to the libcbm compiled library
        yield_table (pandas.DataFrame): the parsed SIT yield output
            of :py:func:`libcbm.input.sit.sit_yield_parser.parse`
        classifiers (pandas.DataFrame): the parsed SIT classifiers output
            of :py:func:`libcbm.input.sit.sit_classifier_parser.parse`
        classifier_values (pandas.DataFrame): the parsed SIT classifier values
            output of :py:func:`libcbm.input.sit.sit_classifier_parser.parse`
        age_classes (pandas.DataFrame): the parsed SIT age classes
            output of :py:func:`libcbm.input.sit.sit_age_class_parser.parse`

    Returns:
        libcbm.model.cbm.cbm_model.CBM: an initialized CBM instance
    """
    cbm = cbm_factory.create(
        dll_path=dll_path,
        dll_config_factory=cbm_defaults.get_libcbm_configuration_factory(
            db_path),
        cbm_parameters_factory=cbm_defaults.get_cbm_parameters_factory(
            db_path),
        merch_volume_to_biomass_factory=lambda:
            cbm_config.merch_volume_to_biomass_config(
                db_path=db_path,
                merch_volume_curves=get_merch_volumes(
                    yield_table, classifiers, age_classes)),
        classifiers_factory=lambda: get_classifiers(
            classifiers, classifier_values))

    return cbm
