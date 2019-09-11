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


def get_merch_volumes(yield_table, classifiers, classifier_values, age_classes,
                      sit_mapping):
    """Create merchantable volume input for initializing the CBM class
    based on CBM Standard import tool formatted data.

    Args:
        yield_table (pandas.DataFrame): the parsed SIT yield output
            of :py:func:`libcbm.input.sit.sit_yield_parser.parse`
        classifiers (pandas.DataFrame): the parsed SIT classifiers output
            of :py:func:`libcbm.input.sit.sit_classifier_parser.parse`
        age_classes (pandas.DataFrame): the parsed SIT age classes
            output of :py:func:`libcbm.input.sit.sit_age_class_parser.parse`
        sit_mapping (libcbm.input.sit.sit_mapping.SITMapping): instance of
            SITMapping used to validate species classifier and fetch species id

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
    yield_table.leading_species = sit_mapping.get_species(
        yield_table.leading_species, classifiers, classifier_values)
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


def initialize_inventory(sit_data, sit_mapping):

    classifier_config = get_classifiers(
        sit_data.classifiers, sit_data.classifier_values)
    classifier_ids = [
        (x["id"], x["name"]) for x in classifier_config["classifiers"]]
    classifier_value_id_lookups = {}

    for identifier, name in classifier_ids:
        classifier_value_id_lookups[name] = {
            x["value"]: x["id"]
            for x in classifier_config["classifier_values"]
            if x["classifier_id"] == identifier}

    classifiers_data = np.column_stack([
        sit_data.inventory[name].map(classifier_value_id_lookups[name])
        for name in list(sit_data.classifiers.name)
    ])

    classifiers_data = np.ascontiguousarray(classifiers_data)
    classifiers_result = pd.DataFrame(
        data=classifiers_data,
        columns=list(sit_data.classifiers.name))

    inventory_result = pd.DataFrame(
        data={
            "age": sit_data.inventory.age,
            "spatial_unit": sit_mapping.get_spatial_unit(
                sit_data.inventory, sit_data.classifiers,
                sit_data.classifier_values),
            "afforestation_pre_type_id": sit_mapping.get_nonforest_cover_ids(
                sit_data.inventory, sit_data.classifiers,
                sit_data.classifier_values),
            "area": sit_data.inventory.area,
            "delay": sit_data.inventory.delay,
            "land_class": sit_mapping.get_land_class_id(
                sit_data.inventory.land_class),
            "historical_disturbance_type":
                sit_mapping.get_disturbance_type_id(
                    sit_data.inventory.historical_disturbance_type),
            "last_pass_disturbance_type":
                sit_mapping.get_disturbance_type_id(
                    sit_data.inventory.last_pass_disturbance_type),
        })
    return classifiers_result, inventory_result


def initialize_cbm(db_path, dll_path, sit_data, sit_mapping):
    """Create an initialized instance of
        :py:class:`libcbm.model.cbm.cbm_model.CBM` based on SIT input

    Args:
        db_path (str): path to a cbm_defaults database
        dll_path (str): path to the libcbm compiled library
        sit_data (object): a standard import dataset as returned by:
            :py:func:`libcbm.input.sit.sit_reader.read`
        sit_mapping (libcbm.input.sit.sit_mapping.SITMapping): class used to
            link sit data to default parameters

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
                    sit_data.yield_table, sit_data.classifiers,
                    sit_data.classifier_values, sit_data.age_classes,
                    sit_mapping)),
        classifiers_factory=lambda: get_classifiers(
            sit_data.classifiers, sit_data.classifier_values))

    return cbm
