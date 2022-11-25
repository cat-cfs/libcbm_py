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
