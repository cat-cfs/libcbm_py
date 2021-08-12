from enum import Enum
import numpy as np
from libcbm import data_helpers
from libcbm.wrapper import libcbm_wrapper_functions
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix_Int


class OperationFormat(Enum):
    MatrixList = 1
    RepeatingCoordinates = 2


class Operation:

    def __init__(self, dll, format, data):
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

    def __init_matrix_list(self, data):
        self.data = data
        self.__matrix_list_p = \
            libcbm_wrapper_functions.get_matrix_list_pointer(self.data)
        self.__matrix_list_len = len(data)

    def __init_repeating(self, data):
        value_len = 1
        for d in data:
            if isinstance(d[2], np.ndarray):
                value_len = d[2].shape[0]
                break
        coordinates = np.array([[x[0], x[1]] for x in data], dtype=int)
        values = np.column_stack(
            [data_helpers.promote_scalar(x[2], size=value_len, dtype=float)
             for x in data])

        self.__repeating_matrix_coords = LibCBM_Matrix_Int(coordinates)
        self.__repeating_matrix_values = LibCBM_Matrix(values)

    def __allocate_op(self, size):
        if self.__op_id is not None:
            self.__dll.free_op(self.__op_id)
        self.__op_id = self.__dll.allocate_op(size)

    def dispose(self):
        if self.__op_id is not None and self.__dll is not None:
            self.__dll.free_op(self.__op_id)
            self.__op_id = None

    def get_op_id(self):
        return self.__op_id

    def set_matrix_index(self, matrix_index):
        if self.format == OperationFormat.MatrixList:
            self.__allocate_op(matrix_index.shape[0])
            self.__dll.handle.call(
                "LibCBM_SetOp", self.__op_id, self.__matrix_list_p,
                self.__matrix_list_len, matrix_index,
                matrix_index.shape[0], 1)
        elif self.format == OperationFormat.RepeatingCoordinates:
            self.__allocate_op(matrix_index.shape[0])
            self.__dll.handle.call(
                "LibCBM_SetOp2", self.__op_id, self.__repeating_matrix_coords,
                self.__repeating_matrix_values, matrix_index,
                matrix_index.shape[0], 1)


def compute(dll, pools, operations, op_processes=None,
            flux=None, enabled=None):
    """Compute pool flows and optionally track the fluxes

    see the methods in :py:class:`libcbm.wrapper.libcbm_wrapper.LibCBMWrapper`

    Args:
        dll (LibCBMWrapper): instance of libcbm wrapper
        pools (pandas.DataFrame): pools dataframe (stands by pools)
        ops (list): list of
            :py:class:`libcbm.wrapper.libcbm_operation.Operation`
        op_processes (iterable, optional): flux indicator op processes.
            Required if flux arg is specified. Defaults to None.
        flux (pandas.DataFrame, optional): Flux indicators dataframe
            (stands by flux-indicator). If not specified, no fluxes are
            tracked. Defaults to None.
        enabled (numpy.ndarray, optional): Flag array of length n-stands
            indicating whether or not to include corresponding rows in
            computation. If set to None, all records are included.
            Defaults to None.
    """

    op_ids = [x.get_op_id() for x in operations]
    if flux is not None:
        dll.compute_flux(op_ids, op_processes, pools, flux, enabled)
    else:
        dll.compute_pools(op_ids, pools, enabled)
