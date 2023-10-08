import numpy as np
import pandas as pd
from libcbm.model.model_definition.model import CBMModel
from libcbm.wrapper.libcbm_operation import Operation
from libcbm.model.model_definition.model_variables import ModelVariables


class ModelMatrixOps:
    def __init__(self, model: CBMModel, op_process_ids: dict[str, int]):
        self._model = model
        self._op_process_ids = op_process_ids
        self._ops_by_name: dict[str, Operation] = {}
        self._op_index_by_name: dict[str, pd.DataFrame] = {}

    def init_operation(
        self,
        name: str,
        op_process: str,
        operation_data: pd.DataFrame,
        model_variables: ModelVariables,
        init_value: int = 1,
    ) -> Operation:
        if name in self._ops_by_name:
            self._ops_by_name[name].dispose()
            del self._ops_by_name[name]
            del self._op_index_by_name[name]

        self._model.pool_names
        op_cols = list(operation_data.columns)
        pool_src_sink_tuples: list[tuple] = [
            tuple(x.split(".")) for x in operation_data.columns
        ]
        matrices = [
            [p[0], p[1], operation_data[op_cols[i]]]
            for i, p in enumerate(pool_src_sink_tuples)
        ]
        self._op_index_by_name[name] = pd.DataFrame(
            index=operation_data.index,
            data={
                "matrix_index": np.arange(
                    0, len(operation_data.index), dtype="uintp"
                )
            },
        )
        matrix_index = self.compute_matrix_index(name, model_variables)
        op = self._model.create_operation(
            matrices,
            "repeating_coordinates",
            self._op_process_ids[op_process],
            matrix_index,
            init_value=init_value,
        )
        self._ops_by_name[name] = op

        return op

    def compute_matrix_index(
        self, op_name: str, model_variables: ModelVariables
    ) -> np.ndarray:
        n_rows = model_variables["pools"].n_rows
        op_index = self._op_index_by_name[op_name]
        if len(op_index.index.names) == 1 and not op_index.index.name[0]:
            if len(op_index.index) == 1:
                # project the single operation to the entire landscape
                return np.full(len(op_index.index), 0, dtype="uintp")
            elif len(op_index.index) == n_rows:
                # there is one operation for each simulation area
                return np.arange(0, len(op_index.index), dtype="uintp")

        merge_data = {}
        for idx_name in op_index.index.names:
            if idx_name == "row_idx":
                merge_data["row_idx"] = np.arange(0, n_rows)
            else:
                s = idx_name.split(".")
                merge_data[idx_name] = model_variables[s[0]][s[1]].to_numpy()

        merge_df = pd.DataFrame(
            merge_data,
        )
        merge_df.index.names = op_index.index.names

        merged = merge_df.merge(
            op_index,
            left_on=op_index.index.names,
            right_index=True,
            how="left",
        )
        return merged["matrix_index"].to_numpy(dtype="uintp")

    def reindex_operation(
        self, name: str, model_variables: ModelVariables
    ) -> Operation:
        matrix_index = self.compute_matrix_index(name, model_variables)
        op = self._ops_by_name[name]
        op.update_index(matrix_index)
        return op
