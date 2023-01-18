from __future__ import annotations
from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm.storage import series
from libcbm.storage import dataframe
from libcbm.storage.dataframe import DataFrame


class ModelOutputProcessor:
    """
    Stores results by timestep using the libcbm.storage.dataframe.DataFrame
    abstraction.  This is not strictly required for running, as the simulation
    state variables are accessible via standard interfaces such as pandas and
    numpy.

    Note the numpy and pandas DataFrame backends will store information in
    memory limiting the scalability of this method.
    """

    def __init__(self):
        self._results: dict[str, DataFrame] = {}

    def append_results(self, t: int, results: CBMVariables):
        """Append results to the output processor.  Values from the specified
        results will be concatenated with previous timestep results.

        Two columns will be added to the internally stored dataframes to
        identify rows: identifier and timestep.

        Args:
            t (int): the timestep
            results (CBMVariables): collection of cbm variables and state for
                the timestep.
        """
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
        """Return the collection of accumulated results

        Returns:
            dict[str, DataFrame]: collection of dataframes holding results.
        """
        return CBMVariables(self._results)
