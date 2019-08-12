import os
import ctypes
from numpy.ctypeslib import ndpointer

from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix_Int
from libcbm.wrapper.libcbm_error import LibCBM_Error
from libcbm.wrapper.cbm import cbm_ctypes

class LibCBM_ctypes():
    """Wrapper for the core low level functions in the libcbm C/C++ library

    Args:
        dll_path (str): path to the compiled LibCBM dll on Windows,
        or compiled LibCBM .so file for Linux
    """
    def __init__(self, dll_path):
        self.handle = False

        cwd = os.getcwd()
        os.chdir(os.path.dirname(dll_path))
        self._dll = ctypes.CDLL(dll_path)
        os.chdir(cwd)
        self.err = LibCBM_Error()

        self._dll.LibCBM_Free.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error structure
            ctypes.c_void_p  # handle pointer
        )

        self._dll.LibCBM_Initialize.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error structure
            ctypes.c_char_p  # config json string
        )
        self._dll.LibCBM_Initialize.restype = ctypes.c_void_p

        self._dll.LibCBM_Allocate_Op.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error structure
            ctypes.c_void_p,  # handle
            ctypes.c_size_t  # n ops
        )
        self._dll.LibCBM_Allocate_Op.restype = ctypes.c_size_t

        self._dll.LibCBM_Free_Op.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error structure
            ctypes.c_void_p,  # handle
            ctypes.c_size_t  # op id
        )

        self._dll.LibCBM_SetOp.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error structure
            ctypes.c_void_p,  # handle
            ctypes.c_size_t,  # op_id
            ctypes.POINTER(LibCBM_Matrix),  # matrices
            ctypes.c_size_t,  # n_matrices
            ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"),  # matrix_index
            ctypes.c_size_t  # n_matrix_index
        )

        self._dll.LibCBM_ComputePools.argtypes = (
                ctypes.POINTER(LibCBM_Error),  # error structure
                ctypes.c_void_p,  # handle
                ctypes.POINTER(ctypes.c_size_t),  # op ids
                ctypes.c_size_t,  # number of op ids
                LibCBM_Matrix,  # pools
                ctypes.POINTER(ctypes.c_int)  # enabled
            )

        self._dll.LibCBM_ComputeFlux.argtypes = (
                ctypes.POINTER(LibCBM_Error),  # error structure
                ctypes.c_void_p,  # handle
                ctypes.POINTER(ctypes.c_size_t),  # op ids
                ctypes.POINTER(ctypes.c_size_t),  # op process ids
                ctypes.c_size_t,  # number of ops
                LibCBM_Matrix,  # pools (n_stands by n_pools)
                LibCBM_Matrix,  # flux (n_stands by n_flux_indicators)
                ctypes.POINTER(ctypes.c_int)  # enabled
            )

        cbm_ctypes.initialize_CBM_ctypes(self._dll)
