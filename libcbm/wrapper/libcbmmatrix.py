import ctypes
import numpy as np


class LibCBM_Matrix(ctypes.Structure):
    """Wrapper for low level C/C++ LibCBM structure of the same name.
    Used to pass a 2 dimensional numpy matrix or single valued array as raw
    float64 memory to LibCBM.
    """
    _fields_ = [('rows', ctypes.c_ssize_t),
                ('cols', ctypes.c_ssize_t),
                ('values', ctypes.POINTER(ctypes.c_double))]

    def __init__(self, matrix):
        if len(matrix.shape) == 1 and matrix.shape[0] == 1:
            self.rows = 1
            self.cols = 1
        elif len(matrix.shape) == 2:
            self.rows = matrix.shape[0]
            self.cols = matrix.shape[1]
        else:
            raise ValueError(
                "matrix must have either 2 dimensions or be a single cell "
                "matrix")
        if not matrix.flags["C_CONTIGUOUS"] or \
           not matrix.dtype == np.double:
            raise ValueError(
                "matrix must be c contiguous and of type np.double")
        self.values = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class LibCBM_Matrix_Int(ctypes.Structure):
    """Wrapper for low level C/C++ LibCBM structure of the same name.
    Used to pass a 2 dimensional numpy matrix or single valued array as raw
    int32 memory to LibCBM.
    """
    _fields_ = [('rows', ctypes.c_ssize_t),
                ('cols', ctypes.c_ssize_t),
                ('values', ctypes.POINTER(ctypes.c_int))]

    def __init__(self, matrix):
        if len(matrix.shape) == 1 and matrix.shape[0] == 1:
            self.rows = 1
            self.cols = 1
        elif len(matrix.shape) == 2:
            self.rows = matrix.shape[0]
            self.cols = matrix.shape[1]
        else:
            raise ValueError(
                "matrix must have either 2 dimensions or be a single cell "
                "matrix")
        if not matrix.flags["C_CONTIGUOUS"] or not matrix.dtype == np.int32:
            raise ValueError(
                "matrix must be c contiguous and of type np.int32")
        self.values = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_int))