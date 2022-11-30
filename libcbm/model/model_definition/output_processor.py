from typing import Callable
from libcbm.storage import series
from libcbm.storage import dataframe
from libcbm.storage.dataframe import DataFrame


class ModelOutputProcessor:
    """
    stores results by timestep using the libcbm.storage.dataframe.DataFrame
    abstraction
    """

    def __init__(
        self,
        output_dataframe_converter: Callable[[DataFrame], DataFrame] = None,
    ):
        self._output_dataframe_converter = output_dataframe_converter
        self._results: dict[str, DataFrame] = {}

    def append_results(self, t: int, results: dict[str, DataFrame]):
        for name, df in results.items():
            if self._output_dataframe_converter:
                results_t = self._output_dataframe_converter(df)
            else:
                results_t = df.copy()

            results_t.add_column(
                series.allocate(
                    "timestep",
                    results_t.n_rows,
                    t,
                    "int32",
                    results_t.backend_type,
                ),
                0,
            )
            if name not in self._results:
                self._results[name] = results_t
            else:
                self._results[name] = dataframe.concat_data_frame(
                    [self._results[name], results_t]
                )

    def get_results(self) -> dict[str, DataFrame]:
        return {k: v for k, v in self._results.items()}
