import pandas as pd
from typing import Union
from libcbm.model.model_definition.model_handle import ModelHandle
from libcbm.wrapper.libcbm_operation import Operation
from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.model_definition.matrix_merge_index import MatrixMergeIndex


def prepare_operation_dataframe(
    df: pd.DataFrame, pool_names: set[str]
) -> pd.DataFrame:
    """Validate and prepare a dataframe formatted to store an indexed matrix
    on every row of the dataframe.


    Args:
        df (pd.DataFrame): the formatted dataframe
        pool_names (set[str]): the pool names for validation of formatted
            column names

    Raises:
        ValueError: index columns names were not formatted as expected
        ValueError: pool source sink column names were not formatted as
            expected, or failed pool name validation

    Returns:
        pd.DataFrame:
    """
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


def init_index(operation_data: pd.DataFrame) -> MatrixMergeIndex:
    if (
        len(operation_data.index.names) == 1
        and not operation_data.index.names[0]
    ):
        return MatrixMergeIndex(len(operation_data.index), None)
    key_data = {}
    names = operation_data.index.names
    df = operation_data.reset_index()
    key_data = {name: df[name].to_numpy() for name in names}
    return MatrixMergeIndex(
        len(df.index),
        key_data,
    )


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
        self._index_len = len(self._operation_data.index)
        self._non_indexed = False
        self._op_index = init_index(self._operation_data)
        self._requires_reindexing = requires_reindexing
        self._init_value = init_value
        self._default_matrix_index = default_matrix_index
        self._op: Union[Operation, None] = None

    def dispose(self):
        if self._op:
            self._op.dispose()

    def get_operation(self, model_variables: ModelVariables) -> Operation:
        if self._op is not None:
            n_rows = model_variables["pools"].n_rows
            curr_idx_len = self._index_len
            must_index = curr_idx_len != 1 or curr_idx_len != n_rows
            if self._requires_reindexing:
                matrix_index = self._op_index.compute_matrix_index(
                    model_variables, self._default_matrix_index
                )
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

        matrix_index = self._op_index.compute_matrix_index(
            model_variables, self._default_matrix_index
        )
        self._op = self._model_handle.create_operation(
            matrices,
            "repeating_coordinates",
            self._op_process_id,
            matrix_index,
            init_value=self._init_value,
        )

        return self._op


class ModelMatrixOps:
    """
    class for managing C flow matrices using a formatted dataframe storage
    scheme.
    """

    def __init__(
        self,
        model_handel: ModelHandle,
        pool_names: list[str],
        op_process_ids: dict[str, int],
    ):
        """Create ModelMatrixOps

        Args:
            model_handel (ModelHandle): the model handle for applying C flows
            pool_names (list[str]): names of pools in the model
            op_process_ids (dict[str, int]): dictionary of op_process_ids for
                categorization of fluxes extracted from the C flows
        """
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
        """Create a C flow operation using a dataframe formatted so that each
            row is an index C flow matrix.

        Args:
            name (str): The operations unique name. If an existing operation
                is stored in this instance it will be overwritten.
            op_process_name (str): The op process name used to categorize
                resulting C fluxes
            op_data (pd.DataFrame): the formatted dataframe containing indexed
                rows of matrices
            requires_reindexing (bool, optional): If set to True each time
                :py:func:`get_operation` is called the matrices are re-indexed
                according to the current simulation state. This is needed for
                example if timestep varying values are involved in the index.
                Defaults to True.
            init_value (int, optional): The default value set on the diagonal
                of each matrix. Diagonal values specified in the parameters
                will overwrite this default. Defaults to 1.
            default_matrix_index (Union[int, None], optional): If simulation
                values are not found within the specified op_data, the
                specified 0 based index will be used as a default fill-value.
                If this value is not specified, such missing values will
                instead result in an error being raised. Defaults to None.
        """
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
        """Get C flow operations for computation of C flows on the current
        model state stored in model_variables.

        Args:
            op_names (list[str]): the sequential names of the stored
                operations to apply (duplicates allowed)
            model_variables (ModelVariables): the current model state:
                pools, flux, state etc.

        Returns:
            list[Operation]: a list of C flow operations to apply
        """
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
