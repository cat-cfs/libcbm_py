import pandas as pd
from libcbm.model.cbm import cbm_defaults
from libcbm.model.cbm import cbm_factory
from libcbm.model.cbm import cbm_config
from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference


def get_classifiers(classifiers, classifier_values):
    classifiers_config = []
    for _, row in classifiers.iterrows():
        values = classifier_values[
            classifier_values.classifier_id == row.id].name
        classifiers_config.append(cbm_config.classifier(
            name=row["name"],
            values=[cbm_config.classifier_value(value=x) for x in list(values)]
        ))

    config = cbm_config.classifier_config(classifiers_config)
    print(config)
    return config


def get_merch_volumes(yield_table, classifiers, age_classes,
                      cbm_defaults_ref):

    default_species_map = {
        x["species_name"]: x["species_id"]
        for x in cbm_defaults_ref.get_species()}

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
            sit_species = match_row["leading_species"]
            if sit_species not in default_species_map:
                raise ValueError(
                    f"species {sit_species} not defined in default species "
                    "map.")
            vols = match_row.iloc[len(classifiers)+1:]
            merch_vols.append({
                "species_id": default_species_map[sit_species],
                "age_volume_pairs": [
                    (ages[i], vols[i]) for i in range(len(vols))]
            })
        output.append(
            cbm_config.merch_volume_curve(
                classifier_set=list(row),
                merch_volumes=merch_vols
            ))
    return output


def initialize_cbm(db_path, dll_path, yield_table, classifiers,
                   classifier_values, age_classes):

    cbm_defaults_ref = CBMDefaultsReference(db_path)

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
                    yield_table, classifiers, age_classes, cbm_defaults_ref)),
        classifiers_factory=lambda: get_classifiers(
            classifiers, classifier_values))

    return cbm
