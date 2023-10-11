import numpy as np
import pandas as pd
from typing import Union
from libcbm.model.model_definition.model_handle import ModelHandle
from libcbm.wrapper.libcbm_operation import Operation
from libcbm.model.model_definition.model_variables import ModelVariables


def prepare_operation_dataframe(
    df: pd.DataFrame, pool_names: set[str]
) -> pd.DataFrame:
    result_df = df.copy()
    cols: list[str] = [s.strip() for s in df.columns]

    # any columns with [colname] are index columns and should be
    # set into the index with the brackets stripped
    index_cols = [c for c in cols if c.startswith("[")]
    if len(index_cols) > 0:
        index_names = []
        for idx_col in index_cols:
            idx_col_err = False
            if not idx_col.endswith("]"):
                idx_col_err = True
            idx_col_unpacked = idx_col[1:-1]
            if idx_col_unpacked != "row_idx":
                idx_col_tokens = idx_col_unpacked.split(".")
                if len(idx_col_tokens) != 2:
                    idx_col_err = True
                if (
                    not idx_col_tokens[0].isidentifier()
                    or not idx_col_tokens[1].isidentifier()
                ):
                    idx_col_err = True
            index_names.append(idx_col_unpacked)
            if idx_col_err:
                raise ValueError(f"unexpected colname format {idx_col}")

        result_df.rename(
            columns={
                i_orig: index_names[i] for i, i_orig in enumerate(index_cols)
            },
            inplace=True,
        )
        # using verify_integrity, will ensure no duplicates
        result_df.set_index(index_names, inplace=True, verify_integrity=True)
    else:
        result_df.index.name = None
    # all other columns are pool source sink pair columns whose name are
    # of the form source.sink
    src_sink_cols = [c for c in result_df.columns]
    for src_sink_name in src_sink_cols:
        src_sink_tokens = src_sink_name.split(".")
        pool_col_error = False
        if len(src_sink_tokens) != 2:
            pool_col_error = True
        if (
            src_sink_tokens[0] not in pool_names
            or src_sink_tokens[1] not in pool_names
        ):
            pool_col_error = True
        if pool_col_error:
            raise ValueError(
                f"unexpected pool source sink column {src_sink_name}"
            )
    return result_df


class OperationWrapper:
    def __init__(
        self,
        name: str,
        model_handle: ModelHandle,
        pool_names: set[str],
        op_process_id: int,
        operation_data: pd.DataFrame,
        requires_reindexing: bool = True,
        init_value: int = 1,
        default_matrix_index: Union[int, None] = None,
    ):
        self._name = name
        self._model_handle = model_handle
        self._op_process_id = op_process_id
        self._operation_data = prepare_operation_dataframe(
            operation_data, pool_names
        )
        self._op_index = self._init_index()
        self._requires_reindexing = requires_reindexing
        self._init_value = init_value
        self._default_matrix_index = default_matrix_index
        self._op: Union[Operation, None] = None

    def _init_index(self) -> pd.DataFrame:
        return pd.DataFrame(
            index=self._operation_data.index,
            data={
                "matrix_index": np.arange(
                    0, len(self._operation_data.index), dtype="uintp"
                )
            },
        )

    def dispose(self):
        if self._op:
            self._op.dispose()

    def get_operation(self, model_variables: ModelVariables) -> Operation:
        if self._op is not None:
            n_rows = model_variables["pools"].n_rows
            curr_idx_len = len(self._op_index.index)
            must_index = curr_idx_len != 1 or curr_idx_len != n_rows
            if self._requires_reindexing:
                matrix_index = self._compute_matrix_index(model_variables)
                self._op.update_index(matrix_index)
                return self._op
            elif not must_index:
                return self._op
            else:
                self._op.dispose()
                self._op = None

        op_cols = list(self._operation_data.columns)
        pool_src_sink_tuples: list[tuple] = [
            tuple(x.split(".")) for x in self._operation_data.columns
        ]
        matrices = [
            [p[0], p[1], self._operation_data[op_cols[i]].to_numpy()]
            for i, p in enumerate(pool_src_sink_tuples)
        ]

        matrix_index = self._compute_matrix_index(model_variables)
        self._op = self._model_handle.create_operation(
            matrices,
            "repeating_coordinates",
            self._op_process_id,
            matrix_index,
            init_value=self._init_value,
        )

        return self._op

    def _compute_matrix_index(
        self, model_variables: ModelVariables
    ) -> np.ndarray:
        n_rows = model_variables["pools"].n_rows

        if (
            len(self._op_index.index.names) == 1
            and not self._op_index.index.names[0]
        ):
            if len(self._op_index.index) == 1:
                # project the single operation to the entire landscape
                return np.full(n_rows, 0, dtype="uintp")
            elif len(self._op_index.index) == n_rows:
                # there is one operation for each simulation area
                return np.arange(0, n_rows, dtype="uintp")
            else:
                raise ValueError(
                    "index length must match model_variables length, or be "
                    "of length 1."
                )

        merge_data = {}
        for idx_name in self._op_index.index.names:
            if idx_name == "row_idx":
                merge_data["row_idx"] = np.arange(0, n_rows)
            else:
                s = idx_name.split(".")
                merge_data[idx_name] = model_variables[s[0]][s[1]].to_numpy()

        merge_df = pd.DataFrame(
            merge_data,
        )

        merged = merge_df.merge(
            self._op_index,
            left_on=self._op_index.index.names,
            right_index=True,
            how="left",
            copy=False,
        )
        matrix_index = merged["matrix_index"]
        is_null_merge_loc = matrix_index.isnull()
        if is_null_merge_loc.any():
            if self._default_matrix_index is not None:
                matrix_index.fillna(
                    self._default_matrix_index,
                    inplace=True)
                return matrix_index.to_numpy(dtype="uintp")
            else:
                raise ValueError(
                    "operation could not be merged due to missing values: "
                    f"{merged[is_null_merge_loc.head()]}"
                )
        return matrix_index.to_numpy()


class ModelMatrixOps:
    def __init__(
        self,
        model_handel: ModelHandle,
        pool_names: list[str],
        op_process_ids: dict[str, int],
    ):
        self._op_wrappers: dict[str, OperationWrapper] = {}
        self._model_handle = model_handel
        self._pool_names = set(pool_names)
        self._op_process_ids = op_process_ids

    def create_operation(
        self,
        name: str,
        op_process_name: str,
        op_data: pd.DataFrame,
        requires_reindexing: bool = True,
        init_value: int = 1,
        default_matrix_index: Union[int, None] = None,
    ):
        if name in self._op_wrappers:
            self._op_wrappers[name].dispose()
            del self._op_wrappers[name]
        self._op_wrappers[name] = OperationWrapper(
            name,
            self._model_handle,
            self._pool_names,
            self._op_process_ids[op_process_name],
            op_data,
            requires_reindexing,
            init_value,
            default_matrix_index,
        )

    def get_operations(
        self, op_names: list[str], model_variables: ModelVariables
    ) -> list[Operation]:
        unique_ops: dict[str, Operation] = {}
        out: list[Operation] = []
        for name in op_names:
            if name not in unique_ops:
                unique_ops[name] = self._op_wrappers[name].get_operation(
                    model_variables
                )
            out.append(unique_ops[name])
        return out

    def dispose(self):
        for w in self._op_wrappers.values():
            w.dispose()
