"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import ctypes

from numpy.ctypeslib import ndpointer
from libcbm.wrapper.libcbm_error import LibCBM_Error
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix_Int
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix


def initialize_CBM_ctypes(dll):
    """Extends a :py:class:`libcbm.wrapper.libcbm_ctypes.LibCBM_ctypes` object
    by wrapping the CBM specific functions contained in the low level c/c++
    library.

    Args:
        libcbm_ctypes (libcbm.wrapper.libcbm_ctypes.LibCBM_ctypes): An
            instance the core LibCBM ctypes wrapper.

    """
    dll.LibCBM_Initialize_CBM.argtypes = (
        ctypes.POINTER(LibCBM_Error),  # error structure
        ctypes.c_void_p,  # handle
        ctypes.c_char_p  # config json string
        )

    dll.LibCBM_AdvanceStandState.argtypes = (
        ctypes.POINTER(LibCBM_Error),  # error structure
        ctypes.c_void_p,  # handle
        ctypes.c_size_t,  # n stands
        LibCBM_Matrix_Int,  # classifiers
        # spatial_units (length n)
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        # disturbance_types (length n)
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        # reset_age (length n)
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

    dll.LibCBM_EndStep.argtypes = (
        # error structure
        ctypes.POINTER(LibCBM_Error),
        # handle
        ctypes.c_void_p,
        # n stands
        ctypes.c_size_t,
        # enabled
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        # growth enabled
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        # age
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        # regeneration_delay
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        # time_since_last_disturbance
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        # time_since_land_class_change
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")
        )

    dll.LibCBM_InitializeLandState.argtypes = (
        ctypes.POINTER(LibCBM_Error),  # error structure
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

    dll.LibCBM_AdvanceSpinupState.argtypes = (
        ctypes.POINTER(LibCBM_Error),  # error structure
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

    dll.LibCBM_EndSpinupStep.argtypes = (
        ctypes.POINTER(LibCBM_Error),  # error structure
        ctypes.c_void_p,  # handle
        ctypes.c_size_t,  # n stands
        # spinup state code (length n)
        ndpointer(ctypes.c_uint, flags="C_CONTIGUOUS"),
        # disturbance_type (length n)
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        # pools (n stands by n pools)
        LibCBM_Matrix,
        # age (length n stands, return value)
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        # sum of slow pools (length n stands, return value)
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        # growth enabled (length n stands, return value)
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")
        )

    dll.LibCBM_GetMerchVolumeGrowthOps.argtypes = (
        ctypes.POINTER(LibCBM_Error),  # error structure
        ctypes.c_void_p,  # handle
        ctypes.ARRAY(ctypes.c_size_t, 2),  # op_ids
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

    dll.LibCBM_GetTurnoverOps.argtypes = (
        ctypes.POINTER(LibCBM_Error),  # error structure
        ctypes.c_void_p,  # libcbm handle
        ctypes.ARRAY(ctypes.c_size_t, 2),  # op_ids
        ctypes.c_size_t,  # n stands
        # spatial unit id (length n)
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"))

    dll.LibCBM_GetDecayOps.argtypes = (
        ctypes.POINTER(LibCBM_Error),  # error structure
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

    dll.LibCBM_GetDisturbanceOps.argtypes = (
        ctypes.POINTER(LibCBM_Error),  # error structure
        ctypes.c_void_p,  # handle
        ctypes.ARRAY(ctypes.c_size_t, 1),  # op_id
        ctypes.c_size_t,  # n stands
        # spatial unit id (length n_stands)
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        # disturbance type ids (length n_stands)
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
    )
