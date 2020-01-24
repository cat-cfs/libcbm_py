

class SITCBMDefaults():

    def __init__(self, cbm_defaults, cbm_defaults_ref, sit_disturbance_types):
        self.cbm_defaults = cbm_defaults
        self.cbm_defaults_ref = cbm_defaults_ref

    def get_afforestation_pre_types(self):
        return self.cbm_defaults_ref.get_afforestation_pre_types()

    def get_species_id(self, species_name):
        return self.cbm_defaults_ref.get_species_id(species_name)

    def get_species(self):
        pass

    def get_spatial_unit_id(self, admin, eco):
        pass

    def get_spatial_unit(self, default_spuid):
        pass

    def get_land_classes(self):
        pass

    def get_disturbance_type_id(self):
        pass

    def get_configuration_factory(self):
        pass

    def get_parameters_factory(self):
        pass