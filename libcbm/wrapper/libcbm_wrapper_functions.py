from __future__ import annotations
import ctypes
import numpy as np
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix


def get_matrix_list_pointer(
    matrices: list[np.ndarray],
):
    """converts a list of numpy matrices to a pointer to an array of
    LibCBM_Matrix

    Args:
        matrices (list): list of 2d or single value arrays
    """
    matrices_array = (LibCBM_Matrix * len(matrices))()
    for i_matrix, matrix in enumerate(matrices):
        matrices_array[i_matrix] = LibCBM_Matrix(matrix)
    matrices_p = ctypes.cast(matrices_array, ctypes.POINTER(LibCBM_Matrix))
    return matrices_p
