from enum import Enum
from typing import Iterable
import numpy as np

from libcbm.wrapper import libcbm_wrapper_functions
from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix_Int
from libcbm.storage.dataframe import DataFrame
from libcbm.storage.dataframe import Series


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
    def __init__(self, dll: str, format: OperationFormat, data: list):
        self.format = format
        self.__dll = dll
        self.__op_id = None
        self.__matrix_list_p = None
        self.__matrix_list_len = None

        self.__repeating_matrix_coords = None
        self.__repeating_matrix_values = None
        if self.format == OperationFormat.MatrixList:
            self.__init_matrix_list(data)
        elif self.format == OperationFormat.RepeatingCoordinates:
            self.__init_repeating(data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()

    def __init_matrix_list(self, data: list):
        self.__matrix_list = data
        self.__matrix_list_p = (
            libcbm_wrapper_functions.get_matrix_list_pointer(
                self.__matrix_list
            )
        )
        self.__matrix_list_len = len(self.__matrix_list)

    def __init_repeating(self, data: list):
        value_len = 1
        for d in data:
            if isinstance(d[2], np.ndarray):
                value_len = d[2].shape[0]
                break
        coordinates = np.array([[x[0], x[1]] for x in data], dtype=np.int32)
        values = np.column_stack(
            [_promote_scalar(x[2], size=value_len, dtype=float) for x in data]
        )

        self.__repeating_matrix_coords = LibCBM_Matrix_Int(coordinates)
        self.__repeating_matrix_values = LibCBM_Matrix(values)

    def __allocate_op(self, size: int):
        if self.__op_id is not None:
            self.__dll.free_op(self.__op_id)
        self.__op_id = self.__dll.allocate_op(size)

    def dispose(self):
        if self.__op_id is not None and self.__dll is not None:
            self.__dll.free_op(self.__op_id)
            self.__op_id = None

    def get_op_id(self) -> int:
        return self.__op_id

    def set_matrix_index(self, matrix_index: np.ndarray):
        if not matrix_index.dtype == np.uintp:
            matrix_index = matrix_index.astype(np.uintp)
        if self.format == OperationFormat.MatrixList:
            self.__allocate_op(matrix_index.shape[0])
            self.__dll.handle.call(
                "LibCBM_SetOp",
                self.__op_id,
                self.__matrix_list_p,
                self.__matrix_list_len,
                matrix_index,
                matrix_index.shape[0],
                1,
            )
        elif self.format == OperationFormat.RepeatingCoordinates:
            self.__allocate_op(matrix_index.shape[0])
            self.__dll.handle.call(
                "LibCBM_SetOp2",
                self.__op_id,
                self.__repeating_matrix_coords,
                self.__repeating_matrix_values,
                matrix_index,
                matrix_index.shape[0],
                1,
            )


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
