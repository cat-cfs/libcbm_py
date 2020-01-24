import numpy as np


class SITCBMDefaults():

    def __init__(self, sit, cbm_defaults, cbm_defaults_ref):
        self.sit = sit
        self.cbm_defaults = cbm_defaults
        self.cbm_defaults_ref = cbm_defaults_ref
        self.sit_disturbance_type_id_lookup = {
            x["name"]: x["sit_disturbance_type_id"]
            for _, x in sit.sit_data.disturbance_types.iterrows()}
        self.default_disturbance_id_lookup = {
            x["name"]: x[]
            for x in self.cbm_defaults_ref.get_disturbance_types()
        }

    def get_afforestation_pre_types(self):
        return self.cbm_defaults_ref.get_afforestation_pre_types()

    def get_species_id(self, species_name):
        return self.cbm_defaults_ref.get_species_id(species_name)

    def get_species(self):
        return self.cbm_defaults.get_species()

    def get_spatial_unit_id(self, admin, eco):
        return self.cbm_defaults_ref.get_spatial_unit_id(admin, eco)

    def get_spatial_unit(self, default_spuid):
        return self.cbm_defaults_ref.get_spatial_unit(default_spuid)

    def get_land_classes(self):
        return self.cbm_defaults.get_land_classes()

    def get_sit_disturbance_type_id(self, sit_dist_type_name):
        if sit_dist_type_name in self.sit_disturbance_type_id_lookup:
            return self.sit_disturbance_type_id_lookup[sit_dist_type_name]
        else:
            raise KeyError(
                f"Specified sit disturbance type value {sit_dist_type_name} "
                "not mapped.")

    def get_default_disturbance_type_id(self, default_dist_type_name):
        if default_dist_type_name in self.default_disturbance_id_lookup:
            return self.default_disturbance_id_lookup[default_dist_type_name]
        else:
            raise KeyError(
                "Specified default disturbance type name "
                f"{default_dist_type_name} not found.")

    def get_configuration_factory(self):
        pass

    def get_parameters_factory(self):
        pass
