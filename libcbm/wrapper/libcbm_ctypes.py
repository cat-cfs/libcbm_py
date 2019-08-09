import os
import ctypes
from numpy.ctypeslib import ndpointer

from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix_Int
from libcbm.wrapper.libcbm_error import LibCBM_Error


class LibCBM_ctypes():
    """Wrapper for the low level functions in the libcbm C/C++ library

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
            ctypes.POINTER(LibCBM_Error),  # error struct
            ctypes.c_void_p  # handle pointer
        )

        self._dll.LibCBM_Initialize.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error struct
            ctypes.c_char_p  # config json string
        )
        self._dll.LibCBM_Initialize.restype = ctypes.c_void_p

        self._dll.LibCBM_Allocate_Op.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error struct
            ctypes.c_void_p,  # handle
            ctypes.c_size_t  # n ops
        )
        self._dll.LibCBM_Allocate_Op.restype = ctypes.c_size_t

        self._dll.LibCBM_Free_Op.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error struct
            ctypes.c_void_p,  # handle
            ctypes.c_size_t  # op id
        )

        self._dll.LibCBM_SetOp.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error struct
            ctypes.c_void_p,  # handle
            ctypes.c_size_t,  # op_id
            ctypes.POINTER(LibCBM_Matrix),  # matrices
            ctypes.c_size_t,  # n_matrices
            ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"),  # matrix_index
            ctypes.c_size_t  # n_matrix_index
        )

        self._dll.LibCBM_ComputePools.argtypes = (
                ctypes.POINTER(LibCBM_Error),  # error struct
                ctypes.c_void_p,  # handle
                ctypes.POINTER(ctypes.c_size_t),  # op ids
                ctypes.c_size_t,  # number of op ids
                LibCBM_Matrix,  # pools
                ctypes.POINTER(ctypes.c_int)  # enabled
            )

        self._dll.LibCBM_ComputeFlux.argtypes = (
                ctypes.POINTER(LibCBM_Error),  # error struct
                ctypes.c_void_p,  # handle
                ctypes.POINTER(ctypes.c_size_t),  # op ids
                ctypes.POINTER(ctypes.c_size_t),  # op process ids
                ctypes.c_size_t,  # number of ops
                LibCBM_Matrix,  # pools (nstands by npools)
                LibCBM_Matrix,  # flux (nstands by nfluxIndicators)
                ctypes.POINTER(ctypes.c_int)  # enabled
            )

        self._dll.LibCBM_Initialize_CBM.argtypes = (
                ctypes.POINTER(LibCBM_Error),  # error struct
                ctypes.c_void_p,  # handle
                ctypes.c_char_p  # config json string
            )

        self._dll.LibCBM_AdvanceStandState.argtypes = (
                ctypes.POINTER(LibCBM_Error),  # error struct
                ctypes.c_void_p,  # handle
                ctypes.c_size_t,  # n stands
                LibCBM_Matrix_Int,  # classifiers
                # spatial_units (length n)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                # disturbance_types (length n)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                # transition_rule_ids (length n)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                # last_disturbance_type (length n) (return value)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                # time_since_last_disturbance (length n) (return value)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                # time_since_land_class_change (length n) (return value)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                # growth_enabled (length n) (return value)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                # enabled (length n) (return value)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                # land_class (length n) (return value)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                # regeneration_delay (length n) (return value)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                # age (length n) (return value)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            )

        self._dll.LibCBM_EndStep.argtypes = (
            # error struct
            ctypes.POINTER(LibCBM_Error),
            # handle
            ctypes.c_void_p,
            # n stands
            ctypes.c_size_t,
            # age
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # regeneration_delay
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # enabled
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")
            )

        self._dll.LibCBM_InitializeLandState.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error struct
            ctypes.c_void_p,  # handle
            ctypes.c_size_t,  # n stands
            # last_pass_disturbance (length n)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # delay (length n)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # initial_age (length n stands)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # spatial unit id (length n stands)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # afforestation pre type id (length n stands)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # pools (n stands by n pools) (return value)
            LibCBM_Matrix,
            # last_disturbance_type (length n stands, return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # time_since_last_disturbance (length n stands, return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # time_since_land_class_change (length n stands, return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # growth_enabled (length n stands, return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # enabled (length n stands, return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # land_class (length n stands, return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # age (length n stands, return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")
        )

        self._dll.LibCBM_AdvanceSpinupState.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error struct
            ctypes.c_void_p,  # handle
            ctypes.c_size_t,  # n stands
            # spatial unit id (nullable, length n_stands)
            ctypes.POINTER(ctypes.c_int),
            # return interval (nullable, length n_stands)
            ctypes.POINTER(ctypes.c_int),
            # minRotations (nullable, length n stands)
            ctypes.POINTER(ctypes.c_int),
            # maxRotations (nullable, length n stands)
            ctypes.POINTER(ctypes.c_int),
            # final age (length n stands)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # delay (length n stands)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # slowpools (length n stands)
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            # historical disturbance type (length n stands)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # last pass disturbance type (length n stands)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # afforestation pre type id (length n stands)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # spinup state code (length n stands, return value)
            ndpointer(ctypes.c_uint, flags="C_CONTIGUOUS"),
            # disturbance type  (length n stands, return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # Rotation num (length n stands, return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # simulation step (length n stands, return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # last rotation slow (length n stands, return value)
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            # enabled (length n stands, return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")
        )

        self._dll.LibCBM_EndSpinupStep.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error struct
            ctypes.c_void_p,  # handle
            ctypes.c_size_t,  # n stands
            # spinup state code (length n)
            ndpointer(ctypes.c_uint, flags="C_CONTIGUOUS"),
            # pools (n stands by n pools)
            LibCBM_Matrix,
            # disturbance_type (length n)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # age (length n stands, return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # sum of slow pools (length n stands, return value)
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            # growth enabled (length n stands, return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")
            )

        self._dll.LibCBM_GetMerchVolumeGrowthOps.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error struct
            ctypes.c_void_p,  # handle
            ctypes.ARRAY(ctypes.c_size_t, 1),  # op_ids
            ctypes.c_size_t,  # n stands
            LibCBM_Matrix_Int,  # classifier values (n stands by n classifiers)
            LibCBM_Matrix,  # pools (n stands by n pools)
            # stand ages (length n stands)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # spatial unit id (length n stands)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # (nullable) last disturbance type (length n stands)
            ctypes.POINTER(ctypes.c_int),
            # (nullable) time since last disturbance (length n stands)
            ctypes.POINTER(ctypes.c_int),
            # (nullable) growth multiplier (length n stands)
            ctypes.POINTER(ctypes.c_double),
            # (nullable) growth enabled (length n stands)
            ctypes.POINTER(ctypes.c_int)
            )

        self._dll.LibCBM_GetTurnoverOps.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error struct
            ctypes.c_void_p,  # libcbm handle
            ctypes.ARRAY(ctypes.c_size_t, 2),  # op_ids
            ctypes.c_size_t,  # n stands
            # spatial unit id (length n)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"))

        self._dll.LibCBM_GetDecayOps.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error struct
            ctypes.c_void_p,  # handle
            ctypes.ARRAY(ctypes.c_size_t, 3),  # op_ids
            ctypes.c_size_t,  # n stands
            # spatial unit id (nullable, length n_stands)
            ctypes.POINTER(ctypes.c_int),
            # use historical mean annual temperature (scalar)
            ctypes.c_bool,
            # mean annual temp (nullable, length n_stands)
            ctypes.POINTER(ctypes.c_double)
            )

        self._dll.LibCBM_GetDisturbanceOps.argtypes = (
            ctypes.POINTER(LibCBM_Error),  # error struct
            ctypes.c_void_p,  # handle
            ctypes.ARRAY(ctypes.c_size_t, 1),  # op_id
            ctypes.c_size_t,  # n stands
            # spatial unit id (length n_stands)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            # disturbance type ids (length n_stands)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            )
