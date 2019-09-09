import pandas as pd
import numpy as np


class SITMapping():

    def __init__(self, config, cbm_defaults_ref):
        self.config = config
        self.cbm_defaults_ref = cbm_defaults_ref

    def get_species(self, species, classifiers, classifier_values):
        merged_classifiers = classifiers.merge(
            classifier_values, left_on="id", right_on="classifier_id",
            suffixes=["_classifier", "_classifier_value"])
        afforestation_pre_types = {
            x["afforestation_pre_type_name"]
            for x in self.cbm_defaults_ref.get_afforestation_pre_types()}

        species_map = {}
        for species_mapping in self.config["species"]["species_mapping"]:
            user_species = species_mapping["user_species"]
            default_species = species_mapping["default_species"]
            if user_species in species_map:
                raise ValueError(
                    f"Specified user species {user_species} mapped multiple "
                    "times.")
            if default_species in afforestation_pre_types:
                # since the species classifier can be mapped to nonforest
                # cover types, but we are not interested in these for the
                # purpose of this function, exclude this from the result,
                # but don't raise an error here.
                continue
            try:
                species_id = self.cbm_defaults_ref.get_species_id(
                    default_species)
            except KeyError:
                raise KeyError(
                    f"mapped default species {default_species} not present in "
                    "default values")
            species_map[user_species] = species_id

        species_classifier = self.config["species"]["species_classifier"]
        species_values = merged_classifiers.loc[
            merged_classifiers["name_classifier"] == species_classifier]

        def get_default_species(species_classifier_value):
            if species_classifier_value in species_map:
                return species_map[species_classifier_value]
            else:
                raise KeyError(
                    f"specified value '{species_classifier_value}' not found "
                    "as key in map of species classifier description to cbm "
                    "default species value.")

        default_species_map = {
            row["classifier_value"]: row["default_species"]
            for _, row in pd.DataFrame({
                "classifier_value": species_values["name_classifier_value"],
                "default_species": species_values["description"].map(
                    get_default_species)}).iterrows()
        }
        # check for values that are defined in the species series but not
        # defined in the default_species_map
        undefined_species = np.setdiff1d(
            species.unique(), list(default_species_map.keys()))
        if len(undefined_species) > 0:
            raise ValueError(
                "Undefined species classifiers (as defined in sit "
                f"classifiers) detected: {undefined_species}"
            )
        return species.map(default_species_map)

    def _get_mapping_error_handling_function(self, sit_map, error_fmt):
        def get_mapped_value(value):
            try:
                return sit_map[value]
            except KeyError:
                raise KeyError(error_fmt.format(value))
        return get_mapped_value

    def _get_spatial_unit_joined_admin_eco(self, inventory, classifiers,
                                           classifier_values):
        merged_classifiers = classifiers.merge(
            classifier_values, left_on="id", right_on="classifier_id",
            suffixes=["_classifier", "_classifier_value"])
        spu_map = {
            x["user_spatial_unit"]: (
                x["default_spatial_unit"]["admin_boundary"],
                x["default_spatial_unit"]["eco_boundary"])
            for x in self.config["spatial_units"]["spu_mapping"]}
        spu_classifier = self.config["spatial_units"]["spu_classifier"]
        spu_values = merged_classifiers.loc[
            merged_classifiers["name_classifier"] == spu_classifier]
        default_spu_map = {
            row["classifier_value"]: row["default_spatial_unit"]
            for _, row in pd.DataFrame({
                "classifier_value": spu_values["name_classifier_value"],
                "default_spatial_unit": spu_values["description"].map(
                    self._get_mapping_error_handling_function(
                        spu_map,
                        error_fmt="specified classifier value description "
                                  "'{}' not found in spatial unit map"))}
            ).iterrows()
        }
        output = []
        for admin, eco in inventory[spu_classifier].map(default_spu_map):
            try:
                output.append(
                    self.cbm_defaults_ref.get_spatial_unit_id(admin, eco))
            except KeyError:
                raise KeyError(
                    "The specified administrative/ecological boundary "
                    f"combination does not exist: '{admin}', '{eco}'")
        return pd.Series(output)

    def _get_spatial_unit_separate_admin_eco(self, inventory, classifiers,
                                             classifier_values):
        merged_classifiers = classifiers.merge(
            classifier_values, left_on="id", right_on="classifier_id",
            suffixes=["_classifier", "_classifier_value"])
        admin_map = {
            x["user_admin_boundary"]: x["default_admin_boundary"]
            for x in self.config["spatial_units"]["admin_mapping"]}
        eco_map = {
            x["user_eco_boundary"]: x["default_eco_boundary"]
            for x in self.config["spatial_units"]["eco_mapping"]}

        admin_classifier = self.config["spatial_units"]["admin_classifier"]
        admin_values = merged_classifiers.loc[
            merged_classifiers["name_classifier"] == admin_classifier]

        default_admin_map = {
            row["classifier_value"]: row["default_admin_boundary"]
            for _, row in pd.DataFrame({
                "classifier_value": admin_values["name_classifier_value"],
                "default_admin_boundary": admin_values["description"].map(
                    self._get_mapping_error_handling_function(
                        admin_map,
                        error_fmt="specified classifier value description "
                                  "'{}' not found in admin boundary map"))
                }).iterrows()
        }
        eco_classifier = self.config["spatial_units"]["eco_classifier"]
        eco_values = merged_classifiers.loc[
            merged_classifiers["name_classifier"] == eco_classifier]
        default_eco_map = {
            row["classifier_value"]: row["default_eco_boundary"]
            for _, row in pd.DataFrame({
                "classifier_value": eco_values["name_classifier_value"],
                "default_eco_boundary": eco_values["description"].map(
                    self._get_mapping_error_handling_function(
                        eco_map,
                        error_fmt="specified classifier value description "
                                  "'{}' not found in ecological boundary map")
                )}).iterrows()
        }

        def spu_map_func(row):
            try:
                return self.cbm_defaults_ref.get_spatial_unit_id(
                    row.default_admin_boundary, row.default_eco_boundary)
            except KeyError:
                raise KeyError(
                    "The specified administrative/ecological boundary "
                    "combination does not exist: "
                    f"'{row.default_admin_boundary}', "
                    f"'{row.default_eco_boundary}'")

        return pd.DataFrame({
            "default_admin_boundary": inventory[admin_classifier].map(
                default_admin_map),
            "default_eco_boundary": inventory[eco_classifier].map(
                default_eco_map)}).apply(spu_map_func, axis=1)

    def get_spatial_unit(self, inventory, classifiers, classifier_values):
        mapping_mode = self.config["spatial_units"]["mapping_mode"]
        if mapping_mode == "SingleDefaultSpatialUnit":
            default_spuid = self.config["spatial_units"]["default_spuid"]
            try:
                self.cbm_defaults_ref.get_spatial_unit(default_spuid)
            except KeyError:
                raise KeyError(
                    "specified spatial unit id not found in defaults: "
                    f"{default_spuid}")

            data = np.ones(inventory.shape[0], dtype=np.int32)
            return pd.Series(data * default_spuid)
        elif mapping_mode == "SeparateAdminEcoClassifiers":
            return self._get_spatial_unit_separate_admin_eco(
                inventory, classifiers, classifier_values)
        elif mapping_mode == "JoinedAdminEcoClassifier":
            return self._get_spatial_unit_joined_admin_eco(
                inventory, classifiers, classifier_values)
        else:
            raise ValueError(
                f"specified mapping_mode is not valid {mapping_mode}")

    def get_nonforest_cover_ids(self, inventory, classifiers,
                                classifier_values):
        non_forest_map = {}
        default_species = {
            x["species_name"] for x in self.cbm_defaults_ref.get_species()}
        default_non_forest = {
            x["afforestation_pre_type_name"]: x["afforestation_pre_type_id"]
            for x in self.cbm_defaults_ref.get_afforestation_pre_types()}

        non_forest_in_species = len(
            set(default_non_forest.keys())
            .intersection([
                x["default_species"]
                for x in self.config["species"]["species_mapping"]])) > 0

        non_forest_classifier = None
        species_classifier = self.config["species"]["species_classifier"]
        if non_forest_in_species:
            non_forest_classifier = species_classifier
            for item in self.config["species"]["species_mapping"]:
                user_value = item["user_species"]
                default_value = item["default_species"]
                if default_value in default_species:
                    non_forest_map[user_value] = -1
                else:
                    try:
                        default_id = default_non_forest[default_value]
                    except KeyError:
                        raise KeyError(
                            "specified default_nonforest_type is not a "
                            f"defined default value {default_value}")
                    non_forest_map[user_value] = default_id

        if "nonforest" in self.config \
                and not self.config["nonforest"] is None \
                and len(self.config["nonforest"]) > 0:

            non_forest_classifier = \
                self.config["nonforest"]["nonforest_classifier"]
            if non_forest_classifier == species_classifier:
                raise ValueError(
                    "single classifier may not be used as both non-forest "
                    "classifier and species classifier")

            if non_forest_in_species:
                raise ValueError(
                    "Nonforest values mapped in species classifier and "
                    "non-forest classifier")
            for item in self.config["nonforest"]["nonforest_mapping"]:
                user_value = item["user_nonforest_type"]
                default_value = item["default_nonforest_type"]
                if user_value in non_forest_map:
                    raise KeyError(
                        "specified user_nonforest_type defined multiple times")
                if default_value is not None:
                    try:
                        default_id = default_non_forest[default_value]
                    except KeyError:
                        raise KeyError(
                            "specified default_nonforest_type is not a "
                            f"defined default value {default_value}")
                    non_forest_map[user_value] = default_id
                else:
                    non_forest_map[user_value] = -1

        merged_classifiers = classifiers.merge(
            classifier_values, left_on="id", right_on="classifier_id",
            suffixes=["_classifier", "_classifier_value"])

        non_forest_classifier_values = merged_classifiers.loc[
            merged_classifiers["name_classifier"] == non_forest_classifier]

        default_nonforest_type_map = {
            row["classifier_value"]: row["default_species"]
            for _, row in pd.DataFrame({
                "classifier_value":
                    non_forest_classifier_values["name_classifier_value"],
                "default_species":
                    non_forest_classifier_values["description"].map(
                        non_forest_map)}).iterrows()
        }
        undefined_values = np.setdiff1d(
            inventory[non_forest_classifier].unique(),
            list(non_forest_map.keys())
        )
        if len(undefined_values) > 0:
            raise ValueError(
                "Undefined non forest classifier values found in inventory "
                f"{undefined_values}")
        return inventory[non_forest_classifier].map(default_nonforest_type_map)

    def get_disturbance_type_id(self, disturbance_type):
        disturbance_type_map = {}
        for item in self.config["disturbance_types"]:
            user_dist_type = item["user_dist_type"]
            default_dist_type = item["default_dist_type"]
            if user_dist_type in disturbance_type_map:
                raise KeyError(
                    f"specified user_dist_type {user_dist_type} appears more "
                    "than one time")
            else:
                dist_type_id = None
                try:
                    dist_type_id = \
                        self.cbm_defaults_ref.get_disturbance_type_id(
                            default_dist_type)
                except KeyError:
                    raise KeyError(
                        "specified mapped default disturbance type not found:"
                        f" {default_dist_type}")
                disturbance_type_map[user_dist_type] = dist_type_id

        def map_func(dist_type):
            if dist_type in disturbance_type_map:
                return disturbance_type_map[dist_type]
            else:
                raise KeyError(
                    f"Specified disturbance type value {dist_type} not "
                    "mapped.")

        return disturbance_type.map(map_func)
