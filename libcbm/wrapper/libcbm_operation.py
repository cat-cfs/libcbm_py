from __future__ import annotations
from enum import Enum
from typing import Iterable
import numpy as np

from libcbm.wrapper import libcbm_wrapper_functions
from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix_Int
from libcbm.storage.dataframe import DataFrame
from libcbm.storage.series import Series


class OperationFormat(Enum):
    MatrixList = 1
    RepeatingCoordinates = 2


def _promote_scalar(value, size, dtype):
    """If the specified value is scalar promote it to a numpy array filled
    with the scalar value, and otherwise return the value.  This is purely
    a helper function to allow scalar parameters for certain vector
    functions

    Args:
        value (numpy.ndarray, number, or None): value to promote
        size (int): the length of the resulting vector if promotion
            occurs
        dtype (object): object used to define the type of the resulting
            vector if promotion occurs


    Returns:
        ndarray or None: returns either the original value, a promoted
            scalar or None depending on the specified values
    """
    if value is None:
        return None
    elif isinstance(value, np.ndarray):
        return value
    else:
        return np.full(shape=size, fill_value=value, dtype=dtype)


class Operation:
    def __init__(
        self,
        dll: LibCBMWrapper,
        format: OperationFormat,
        data: list,
        op_process_id: int,
        matrix_index: np.ndarray,
        init_value: int = 1,
    ):
        self.format = format
        self._dll = dll
        self._op_id = None
        self._matrix_list_p = None
        self._matrix_list_len = None
        self._op_process_id = op_process_id
        self._repeating_matrix_coords = None
        self._repeating_matrix_values = None
        self._init_value = init_value
        if self.format == OperationFormat.MatrixList:
            self._init_matrix_list(data)
        elif self.format == OperationFormat.RepeatingCoordinates:
            self._init_repeating(data)
        self._set_op(matrix_index)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()

    def _init_matrix_list(self, data: list):
        self.__matrix_list = data
        self._matrix_list_p = libcbm_wrapper_functions.get_matrix_list_pointer(
            self.__matrix_list
        )
        self._matrix_list_len = len(self.__matrix_list)

    def _init_repeating(self, data: list):
        value_len = 1
        for d in data:
            if isinstance(d[2], np.ndarray):
                value_len = d[2].shape[0]
                break
        coordinates = np.array([[x[0], x[1]] for x in data], dtype=np.int32)
        values = np.column_stack(
            [_promote_scalar(x[2], size=value_len, dtype=float) for x in data]
        )

        self._repeating_matrix_coords = LibCBM_Matrix_Int(coordinates)
        self._repeating_matrix_values = LibCBM_Matrix(values)

    def _allocate_op(self, size: int):
        if self._op_id is not None:
            self._dll.free_op(self._op_id)
        self._op_id = self._dll.allocate_op(size)

    @property
    def op_process_id(self):
        return self._op_process_id

    def dispose(self):
        if self._op_id is not None and self._dll is not None:
            self._dll.free_op(self._op_id)
            self._op_id = None

    def get_op_id(self) -> int:
        return self._op_id

    def _set_op(self, matrix_index: np.ndarray):
        if not matrix_index.dtype == np.uintp:
            matrix_index = matrix_index.astype(np.uintp)
        if self.format == OperationFormat.MatrixList:
            self._allocate_op(matrix_index.shape[0])
            self._dll.handle.call(
                "LibCBM_SetOp",
                self._op_id,
                self._matrix_list_p,
                self._matrix_list_len,
                matrix_index,
                matrix_index.shape[0],
                self._init_value,
            )
        elif self.format == OperationFormat.RepeatingCoordinates:
            self._allocate_op(matrix_index.shape[0])
            self._dll.handle.call(
                "LibCBM_SetOp2",
                self._op_id,
                self._repeating_matrix_coords,
                self._repeating_matrix_values,
                matrix_index,
                matrix_index.shape[0],
                self._init_value,
            )

    def update_index(self, matrix_index: np.ndarray):
        if not matrix_index.dtype == np.uintp:
            matrix_index = matrix_index.astype(np.uintp)
        self._dll.update_op_index(self._op_id, matrix_index)


def compute(
    dll: LibCBMWrapper,
    pools: DataFrame,
    operations: list[Operation],
    op_processes: Iterable[int] = None,
    flux: DataFrame = None,
    enabled: Series = None,
):
    """Compute pool flows and optionally track the fluxes

    see the methods in :py:class:`libcbm.wrapper.libcbm_wrapper.LibCBMWrapper`

    Args:
        dll (LibCBMWrapper): instance of libcbm wrapper
        pools (DataFrame): pools dataframe (stands by pools)
        ops (list): list of
            :py:class:`libcbm.wrapper.libcbm_operation.Operation`
        op_processes (iterable, optional): flux indicator op processes.
            Required if flux arg is specified. Defaults to None.
        flux (DataFrame, optional): Flux indicators dataframe
            (stands by flux-indicator). If not specified, no fluxes are
            tracked. Defaults to None.
        enabled (Series, optional): Flag array of length n-stands
            indicating whether or not to include corresponding rows in
            computation. If set to None, all records are included.
            Defaults to None.
    """

    op_ids = [x.get_op_id() for x in operations]
    if flux is not None:
        dll.compute_flux(op_ids, op_processes, pools, flux, enabled)
    else:
        dll.compute_pools(op_ids, pools, enabled)
