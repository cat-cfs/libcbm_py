import pandas as pd
import numpy as np


class SITMapping():

    def __init__(self, config, cbm_defaults_ref):
        self.config = config
        self.cbm_defaults_ref = cbm_defaults_ref

        # validations:
        # 1. check for duplicate values in left values of user/default maps
        # 2. check for undefined values in right values of user/default maps

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
                    spu_map)}
            ).iterrows()
        }
        output = []
        for admin, eco in inventory[spu_classifier].map(default_spu_map):
            try:
                output.append(
                    self.cbm_defaults_ref.get_spatial_unit_id(admin,eco))
            except KeyError:
                raise KeyError(
                    "The specified administrative/ecological boundary "
                    f"combination does not exist: '{admin}', '{eco}'")
        pd.Series(output)

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
                    admin_map)}
            ).iterrows()
        }
        eco_classifier = self.config["spatial_units"]["eco_classifier"]
        eco_values = merged_classifiers.loc[
            merged_classifiers["name_classifier"] == eco_classifier]
        default_eco_map = {
            row["classifier_value"]: row["default_eco_boundary"]
            for _, row in pd.DataFrame({
                "classifier_value": eco_values["name_classifier_value"],
                "default_eco_boundary": eco_values["description"].map(eco_map)}
            ).iterrows()
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
