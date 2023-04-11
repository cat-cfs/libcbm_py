from __future__ import annotations
import pandas as pd
from libcbm.input.sit.sit_mapping import SITMapping
from libcbm.model.cbm import cbm_config
from libcbm.input.sit.sit_reader import SITData


class SITIdentifierMapping:
    def __init__(self, sit_data: SITData, sit_mapping: SITMapping):
        sit_disturbance_types = sit_data.disturbance_types.copy()
        sit_disturbance_types.insert(
            0,
            "default_disturbance_type_id",
            sit_mapping.get_default_disturbance_type_id(
                sit_disturbance_types.name
            ),
        )
        # start with the "null" disturbance type/default disturbance type
        self.default_disturbance_name_map: dict[int, str] = {0: ""}
        self.default_disturbance_id_map: dict[int, int] = {0: 0}
        self.disturbance_id_map: dict[int, int] = {0: 0}
        self.disturbance_name_map: dict[int, str] = {0: ""}

        default_dist_type_lookup = (
            sit_mapping.sit_cbm_defaults.default_disturbance_id_lookup
        )
        default_name_lookup = {
            int(v): k for k, v in default_dist_type_lookup.items()
        }
        self.default_disturbance_name_map.update(
            {
                row.sit_disturbance_type_id: default_name_lookup[
                    int(row.default_disturbance_type_id)
                ]
                for _, row in sit_disturbance_types.iterrows()
            }
        )
        self.default_disturbance_id_map.update(
            {
                row.sit_disturbance_type_id: row.default_disturbance_type_id
                for _, row in sit_disturbance_types.iterrows()
            }
        )

        self.disturbance_id_map.update(
            {
                row.sit_disturbance_type_id: row.id
                for _, row in sit_disturbance_types.iterrows()
            }
        )

        self.disturbance_name_map.update(
            {
                row.sit_disturbance_type_id: row["name"]
                for _, row in sit_disturbance_types.iterrows()
            }
        )

        classifiers_config = get_classifiers(
            sit_data.classifiers, sit_data.classifier_values
        )
        idx = cbm_config.get_classifier_indexes(classifiers_config)
        self.classifier_names = idx["classifier_names"]
        self.classifier_ids = idx["classifier_ids"]
        self.classifier_value_ids = idx["classifier_value_ids"]
        self.classifier_value_names = idx["classifier_value_names"]


def get_merch_volumes(
    yield_table: pd.DataFrame,
    classifiers: pd.DataFrame,
    classifier_values: pd.DataFrame,
    age_classes: pd.DataFrame,
    sit_mapping: SITMapping,
) -> list:
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
        list: configuration for CBM. See:
            :py:mod:`libcbm.model.cbm.cbm_config`
    """
    classifier_names: list[str] = list(classifiers["name"])
    unique_csets = (
        yield_table[classifier_names].drop_duplicates().reset_index(drop=True)
    )
    unique_csets.insert(
        0, "yield_group_index", range(0, len(unique_csets.index))
    )

    yield_table_processed = yield_table.copy()
    yield_table_processed["leading_species"] = sit_mapping.get_species(
        yield_table["leading_species"], classifiers, classifier_values
    )
    yield_table_grouped = unique_csets.merge(
        yield_table_processed,
        left_on=classifier_names,
        right_on=classifier_names,
        how="left",
    )

    output_data = {}
    ages = list(age_classes["end_year"])
    for _, row in yield_table_grouped.iterrows():
        out_idx = int(row["yield_group_index"])
        start_range = len(classifiers) + 2
        vols = row.iloc[start_range:]
        merch_vols = {
            "species_id": int(row["leading_species"]),
            "age_volume_pairs": [(ages[i], vols[i]) for i in range(len(vols))],
        }
        if out_idx not in output_data:
            output_data[out_idx] = {
                "classifier_set": [
                    str(row[name]) for name in classifier_names
                ],
                "merch_volumes": [merch_vols],
            }
        else:
            output_data[out_idx]["merch_volumes"].append(merch_vols)
    output = []
    for out_idx, data in output_data.items():
        output.append(
            cbm_config.merch_volume_curve(
                classifier_set=data["classifier_set"],
                merch_volumes=data["merch_volumes"],
            )
        )
    return output


def get_classifiers(
    classifiers: pd.DataFrame, classifier_values: pd.DataFrame
) -> dict[str, list]:
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
            classifier_values.classifier_id == row.id
        ].name
        classifiers_config.append(
            cbm_config.classifier(
                name=row["name"],
                values=[
                    cbm_config.classifier_value(value=x) for x in list(values)
                ],
            )
        )

    config = cbm_config.classifier_config(classifiers_config)
    return config
