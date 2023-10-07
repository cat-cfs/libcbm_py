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
        self._op_data_by_name: dict[str, pd.DataFrame] = {}

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
            del self._op_data_by_name[name]

        self._model.pool_names
        op_cols = list(operation_data.columns)
        pool_src_sink_tuples: list[tuple] = [
            tuple(x.split(".")) for x in operation_data.columns
        ]
        matrices = [
            [p[0], p[1], operation_data[op_cols[i]]]
            for i, p in enumerate(pool_src_sink_tuples)
        ]
        matrix_index = self.compute_matrix_index(
            operation_data, model_variables
        )
        op = self._model.create_operation(
            matrices,
            "repeating_coordinates",
            self._op_process_ids[op_process],
            matrix_index,
            init_value=init_value,
        )
        self._ops_by_name[name] = op
        self._op_data_by_name[name] = operation_data
        return op

    def compute_matrix_index(
        self, operation: pd.DataFrame, model_variables: ModelVariables
    ) -> np.ndarray:
        if len(operation.index.names) == 1 and not operation.index.name[0]:
            if len(operation.index) == 1:
                # project the single operation to the entire landscape
                return np.full(len(operation.index), 0, dtype="uintp")
            elif len(operation.index) == model_variables["pools"].n_rows:
                # there is one operation for each simulation area
                return np.arange(0, len(operation.index), dtype="uintp")
        split_index_names = [s.split(".") for s in operation.index.names]
        merge_data = {
            operation.index.names[i]: model_variables[s[0]][s[1]].to_numpy()
            for i, s in enumerate(split_index_names)
        }
        merge_df = pd.DataFrame(
            merge_data,
        )
        merge_df.index.names = operation.index.names
        operation_merge = pd.DataFrame(
            index=operation.index,
            data=np.arange(0, len(operation.index), dtype="uintp"),
        )
        merged = merge_df.merge(
            operation_merge,
            left_on=operation.index.names,
            right_index=True,
            how="left",
        )
        return merged["matrix_index"].to_numpy(dtype="uintp")

    def reindex_operation(
        self, name: str, model_variables: ModelVariables
    ) -> Operation:
        op = self._ops_by_name[name]
        op_data = self._op_data_by_name[name]
        matrix_index = self.compute_matrix_index(op_data, model_variables)
        op.update_index(matrix_index)
        return op
