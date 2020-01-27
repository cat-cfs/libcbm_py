from libcbm.model.cbm.cbm_defaults_reference import CBMDefaultsReference
from libcbm.model.cbm import cbm_defaults


class SITCBMDefaults(CBMDefaultsReference):

    def __init__(self, sit, db_path, locale_code="en-CA"):
        super().__init__(db_path, locale_code)
        self.sit = sit
        self.db_path = db_path
        self.sit_disturbance_type_id_lookup = {
            x["name"]: x["sit_disturbance_type_id"]
            for _, x in sit.sit_data.disturbance_types.iterrows()}
        self.default_disturbance_id_lookup = {
            x["disturbance_type_name"]: x["disturbance_type_id"]
            for x in self.get_disturbance_types()
        }

    def get_sit_disturbance_type_id(self, sit_dist_type_name):
        if sit_dist_type_name in self.sit_disturbance_type_id_lookup:
            return self.sit_disturbance_type_id_lookup[sit_dist_type_name]
        else:
            raise KeyError(
                f"Specified sit disturbance type value {sit_dist_type_name} "
                "not mapped.")

    def get_configuration_factory(self):
        return cbm_defaults.get_libcbm_configuration_factory(self.db_path)

    def get_parameters_factory(self):
        param_func = cbm_defaults.get_cbm_parameters_factory(self.db_path)
        default_parameters = param_func()

        def factory():
            return default_parameters
        return factory
