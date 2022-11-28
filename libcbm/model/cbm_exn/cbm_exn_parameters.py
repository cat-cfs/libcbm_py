import pandas as pd


class CBMEXNParameters:
    def __init__(self, parameters: dict):
        self._parameters = parameters

    def get_slow_mixing_rate(self) -> float:
        pass

    def get_turnover_parameters(self) -> pd.DataFrame:
        pass

    def get_sw_hw_map(self) -> dict[int, int]:
        """
        returns a map of speciesid: sw_hw where sw_hw is either 0: sw or 1: hw
        """
        pass

    def get_root_parameters(self) -> dict[str, float]:
        pass

    def get_decay_parameter(self, dom_pool: str) -> dict[str, float]:
        pass

    def get_disturbance_matrices() -> pd.DataFrame:
        """
        Gets a dataframe with disturbance matrix value information.

        Columns::

         * disturbance_matrix_id
         * source_pool_id
         * sink_pool_id
         * proportion

        """
        pass

    def get_disturbance_matrix_associations() -> pd.DataFrame:
        """
        Gets a dataframe with disturbance matrix assocation information

        Columns::

         * disturbance_type_id
         * spatial_unit_id
         * sw_hw
         * disturbance_matrix_id

        """
        pass
