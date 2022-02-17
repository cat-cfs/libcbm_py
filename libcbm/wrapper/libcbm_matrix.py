# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import ctypes
import numpy as np


class LibCBM_Matrix_Base(ctypes.Structure):
    """Wrapper for low level C/C++ LibCBM structures LibCBM_Matrix and
    LibCBM_Matrix_Int. Used to pass a 2 dimensional numpy matrix or single
    valued array as raw memory to LibCBM.

    Args:
        matrix (numpy.ndarray): a 2 dimensional numpy array, or single value

    Raises:
        ValueError: matrix must have either 2 dimensions or be a scalar
        ValueError: matrix must be of the correct type
    """

    def __init__(self, matrix, matrix_np_type, matrix_c_type):
        if matrix.size == 1:
            self.rows = 1
            self.cols = 1
        elif len(matrix.shape) == 2:
            self.rows = matrix.shape[0]
            self.cols = matrix.shape[1]
        else:
            raise ValueError(
                "matrix must have either 2 dimensions or be a single cell "
                "matrix"
            )
        if not matrix.dtype == matrix_np_type:
            raise ValueError(
                "matrix must be of type {0}. Got {1}".format(
                    matrix_np_type, matrix.dtype
                )
            )
        if not matrix.flags["C_CONTIGUOUS"]:
            raise ValueError("specified matrix is not C_CONTIGUOUS")
        self.values = matrix.ctypes.data_as(ctypes.POINTER(matrix_c_type))


class LibCBM_Matrix(LibCBM_Matrix_Base):
    """64 bit float specialization of
    :py:class:`libcbm.wrapper.libcbm_matrix.LibCBM_Matrix_Base`

    Args:
        matrix (numpy.ndarray): a 2 dimensional numpy array, or single value
    """

    _matrix_c_type = ctypes.c_double
    _matrix_np_type = np.double

    _fields_ = [
        ("rows", ctypes.c_ssize_t),
        ("cols", ctypes.c_ssize_t),
        ("values", ctypes.POINTER(_matrix_c_type)),
    ]

    def __init__(self, matrix):
        #  hang onto a reference of the matrix to prevent the pointers
        #  becoming invalid
        self.matrix = matrix
        LibCBM_Matrix_Base.__init__(
            self,
            matrix,
            LibCBM_Matrix._matrix_np_type,
            LibCBM_Matrix._matrix_c_type,
        )


class LibCBM_Matrix_Int(LibCBM_Matrix_Base):
    """32 bit int specialization of
    :py:class:`libcbm.wrapper.libcbm_matrix.LibCBM_Matrix_Base`

    Args:
        matrix (numpy.ndarray): a 2 dimensional numpy array, or single value
    """

    _matrix_c_type = ctypes.c_int
    _matrix_np_type = np.int32

    _fields_ = [
        ("rows", ctypes.c_ssize_t),
        ("cols", ctypes.c_ssize_t),
        ("values", ctypes.POINTER(_matrix_c_type)),
    ]

    def __init__(self, matrix):
        #  hang onto a reference of the matrix to prevent the pointers
        #  becoming invalid
        self.matrix = matrix
        LibCBM_Matrix_Base.__init__(
            self,
            matrix,
            LibCBM_Matrix_Int._matrix_np_type,
            LibCBM_Matrix_Int._matrix_c_type,
        )
