import pandas as pd


class CBMEXNParameters:
    def __init__(self, parameters: dict):
        self._parameters = parameters

    def get_slow_mixing_rate(self) -> float:
        pass

    def get_turnover_parameters(self) -> pd.DataFrame:
        pass
