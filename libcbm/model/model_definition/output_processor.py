from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm.storage import series
from libcbm.storage import dataframe
from libcbm.storage.dataframe import DataFrame


class ModelOutputProcessor:
    """
    stores results by timestep using the libcbm.storage.dataframe.DataFrame
    abstraction
    """

    def __init__(self):
        self._results: dict[str, DataFrame] = {}

    def append_results(self, t: int, results: CBMVariables):
        for name, df in results.get_collection().items():
            results_t = df.copy()
            results_t.add_column(
                series.range(
                    "identifier",
                    1,
                    results_t.n_rows + 1,
                    1,
                    "int64",
                    results_t.backend_type,
                ),
                0,
            )
            results_t.add_column(
                series.allocate(
                    "timestep",
                    results_t.n_rows,
                    t,
                    "int32",
                    results_t.backend_type,
                ),
                1,
            )
            if name not in self._results:
                self._results[name] = results_t
            else:
                self._results[name] = dataframe.concat_data_frame(
                    [self._results[name], results_t]
                )

    def get_results(self) -> dict[str, DataFrame]:
        return CBMVariables(self._results)
