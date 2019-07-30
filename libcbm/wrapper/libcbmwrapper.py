import ctypes
import logging
import sqlite3
import os
import numpy as np
from numpy.ctypeslib import ndpointer

from libcbm.wrapper.libcbmmatrix import LibCBM_Matrix
from libcbm.wrapper.libcbmmatrix import LibCBM_Matrix_Int
from libcbm.wrapper.libcbmerror import LibCBM_Error


def getNullableNdarray(a, type=ctypes.c_double):
    """helper method for wrapper parameters that can be specified either as
    null pointers or pointers to numpy memory

    Arguments:
        a {array_like} or {None} -- array to convert to pointer, if None is
        specified None is returned.

    Keyword Arguments:
        type {object} -- type supported by ctypes.POINTER
        (default: {ctypes.c_double})

    Returns:
        None or ctypes.POINTER -- if the specified argument is None, None is
        returned, otherwise the argument is converted to a C_CONTIGUOUS
        pointer to the underlying ndarray data.
    """
    if a is None:
        return None
    else:
        result = np.ascontiguousarray(a).ctypes.data_as(ctypes.POINTER(type))
        return result


class LibCBMWrapper():
    def __init__(self, dllpath):
        """Initializes the underlying LibCBM library, storing the allocated
        handle in this instance.

        Arguments:
            dllpath {str} -- path to the compiled LibCBM dll on Windows,
            or compiled LibCBM .so file for Linux

        Returns:
            None
        """
        self.handle = False

        cwd = os.getcwd()
        os.chdir(os.path.dirname(dllpath))
        self._dll = ctypes.CDLL(dllpath)
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
            # use historic mean annual temperature (scalar)
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """frees the allocated libcbm handle"""
        if self.handle:
            err = LibCBM_Error()
            self._dll.LibCBM_Free(ctypes.byref(err), self.handle)
            if err.Error != 0:
                raise RuntimeError(err.getErrorMessage())

    def Initialize(self, config):
        """Initialize libcbm with pools, and flux indicators

        Arguments:
            config {str} -- a json formatted string containing configuration
            for libcbm pools and flux definitions.

            The number of pools, and flux indicators defined here, corresponds
            to other data dimensions used during the lifetime of this instance:
                1. The number of pools here defines the number of columns in
                   the pool value matrix used by several other libCBM functions
                2. The number of flux_indicators here defines the number of
                   columns in the flux indicator matrix in the
                   ComputeFlux method.

            Example:
                {
                    "pools": [
                        {"id": 1, "index": 0, "name": "pool_1"},
                        {"id": 2, "index": 1, "name": "pool_2"},
                           ...
                        {"id": n, "index": n-1, "name": "pool_n"}],

                    "flux_indicators": [
                        {
                            "id": 1,
                            "index": 0,
                            "process_id": 1,
                            "source_pools": [1, 2]
                            "sink_pools": [3]
                        },
                        {
                            "id": 2,
                            "index": 1,
                            "process_id": 1,
                            "source_pools": [1, 2]
                            "sink_pools": [3]
                        },
                        ...
                    ]
                }

            Pool/Flux Indicators configuration rules:
                1. ids may be any integer, but are constrained to be unique
                   within the set of pools.
                2. indexes must be the ordered set of integers from 0 to
                   n_pools - 1.
                3. For flux indicator source_pools and sink_pools, list values
                   correspond to id values in the collection of pools



        Raises:
            RuntimeError: if an error is detected in libCBM, it will be
            re-raised with an appropriate error message.
        """
        p_config = ctypes.c_char_p(config.encode("UTF-8"))

        self.handle = self._dll.LibCBM_Initialize(
            ctypes.byref(self.err),  # error struct
            p_config
            )

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

    def AllocateOp(self, n):
        """Allocates storage for n matrices, returning an id for the
        allocated block.

        Arguments:
            n {int} -- The number of matrices to allocate.

        Raises:
            AssertionError: raised if the Initialize method was not called
            prior to this method.
            RuntimeError: if an error is detected in libCBM, it will be
            re-raised with an appropriate error message.

        Returns:
            int -- the id for an allocated block of matrices
        """
        if not self.handle:
            raise AssertionError("dll not initialized")

        op_id = self._dll.LibCBM_Allocate_Op(
            ctypes.byref(self.err),
            self.handle, n)

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

        return op_id

    def FreeOp(self, op_id):
        """Deallocates a matrix block that was allocated by the AllocateOp method.

        Arguments:
            op_id {int} -- The id for an allocated block of matrices.

        Raises:
            AssertionError: raised if the Initialize method was not called
                prior to this method.
            RuntimeError: if an error is detected in libCBM, it will be
                re-raised with an appropriate error message.
        """
        if not self.handle:
            raise AssertionError("dll not initialized")

        self._dll.LibCBM_Free_Op(
            ctypes.byref(self.err),
            self.handle, op_id)

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

    def SetOp(self, op_id, matrices, matrix_index):
        """Assigns values to an allocated block of matrices.

        Arguments:
            op_id {int} -- The id for an allocated block of matrices
            matrices {list of ndarray} -- a list of n by 3 matrices which are
                coordinate format triplet values (row,column,value).  All
                defined row/column combinations are set with the value, and
                all other matrix cells are assumed to be 0.
            matrix_index {ndarray} -- an array of length n stands where the
                value is an index to a matrix in the specified list of matrices
                provided to this function.

        Raises:
            AssertionError: raised if the Initialize method was not called
                prior to this method.
            RuntimeError: if an error is detected in libCBM, it will be
                re-raised with an appropriate error message.
        """
        if not self.handle:
            raise AssertionError("dll not initialized")
        matrices_array = (LibCBM_Matrix * len(matrices))()
        for i, x in enumerate(matrices):
            matrices_array[i] = LibCBM_Matrix(x)
        matrices_p = ctypes.cast(matrices_array, ctypes.POINTER(LibCBM_Matrix))
        self._dll.LibCBM_SetOp(
            ctypes.byref(self.err), self.handle, op_id, matrices_p,
            len(matrices), matrix_index, matrix_index.shape[0])
        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

    def ComputePools(self, ops, pools, enabled=None):
        """Computes flows between pool values for all stands.

        Each value in the ops parameter is an id to a matrix block, and is also
        conceptually a list of matrixes of length n stands.

        Performs the following computation:

            for op in ops:
                for s in len(n_stands):
                    M = get_matrix(op,s)
                    pools[s,:] = np.matmul(pools[s,:], M)

        Where get_matrix is a function returning the matrix for the
        op, stand index combination.

        Arguments:
            ops {ndarray} -- list of matrix block ids as allocated by the
                AllocateOp function.
            pools {ndarray} -- matrix of shape n_stands by n_pools. The values
                in this matrix are updated by this function.

        Keyword Arguments:
            enabled {ndarray} -- optional int vector of length n stands. If
                specified, enables or disables flows for each stand, based on
                the value at each stand index. (0 is disabled, !0 is enabled)
                If unspecified, all flows are assumed to be enabled.
                (default: {None})

        Raises:
            AssertionError: raised if the Initialize method was not called
                prior to this method.
            RuntimeError: if an error is detected in libCBM, it will be
                re-raised with an appropriate error message.
        """
        if not self.handle:
            raise AssertionError("dll not initialized")

        n_ops = len(ops)
        poolMat = LibCBM_Matrix(pools)
        ops_p = ctypes.cast(
            (ctypes.c_size_t*n_ops)(*ops), ctypes.POINTER(ctypes.c_size_t))

        self._dll.LibCBM_ComputePools(
            ctypes.byref(self.err), self.handle, ops_p, n_ops, poolMat,
            getNullableNdarray(enabled, type=ctypes.c_int))

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

    def ComputeFlux(self, ops, op_processes, pools, flux, enabled=None):
        """Computes and tracks flows between pool values for all stands.

        Performs the same operation as ComputePools, except that the fluxes
        are tracked in the specified flux parameter, according to the
        flux_indicators configuration passed to the LibCBM initialize method.

        Arguments:
            ops {ndarray} -- list of matrix block ids as allocated by the
                AllocateOp function.
            op_processes {ndarray} -- list of integers of length n_ops.
                Ids referencing flux indicator process_id definition in the
                LibCBM Initialize method.
            pools {ndarray} -- matrix of shape n_stands by n_pools. The values
                in this matrix are updated by this function.
            flux {ndarray} -- matrix of shape n_stands by n_flux_indicators.
                The values in this matrix are updated by this function.

        Keyword Arguments:
            enabled {ndarray} -- optional int vector of length n stands. If
                specified, enables or disables flows for each stand, based on
                the value at each stand index. (0 is disabled, !0 is enabled)
                If unspecified, all flows are assumed to be enabled.
                (default: {None})

        Raises:
            AssertionError: raised if the Initialize method was not called
                prior to this method.
            ValueError: raised when parameters passed to this function are not
                valid.
            RuntimeError: if an error is detected in libCBM, it will be
                re-raised with an appropriate error message.
        """
        if not self.handle:
            raise AssertionError("dll not initialized")

        n_ops = len(ops)
        if len(op_processes) != n_ops:
            raise ValueError("ops and op_processes must be of equal length")
        poolMat = LibCBM_Matrix(pools)
        fluxMat = LibCBM_Matrix(flux)

        ops_p = ctypes.cast(
            (ctypes.c_size_t*n_ops)(*ops), ctypes.POINTER(ctypes.c_size_t))
        op_process_p = ctypes.cast(
            (ctypes.c_size_t*n_ops)(*op_processes),
            ctypes.POINTER(ctypes.c_size_t))

        self._dll.LibCBM_ComputeFlux(
            ctypes.byref(self.err), self.handle, ops_p, op_process_p, n_ops,
            poolMat, fluxMat, getNullableNdarray(enabled, type=ctypes.c_int))

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

    def InitializeCBM(self, config):
        """Initializes CBM-specific functionality within LibCBM

        Arguments:
            config {str} -- A json formatted string containing CBM
                configuration.

            See libcbm.configuration.cbm_defaults for construction of the
            "cbm_defaults" value.  It is too large to include a useful example
            here.

            Example:
                {
                    "cbm_defaults": {"p1": {}, "p2": {}, ..., "pN": {}},
                    "classifiers": [
                        {"id": 1, "name": "a"},
                        {"id": 2, "name": "b"},
                        {"id": 3, "name": "c"}
                    ],
                    "classifier_values": [
                        {
                            "id": 1,
                            "classifier_id": 1,
                            "value": "a1",
                            "description": "a1"
                        },
                        {
                            "id": 2,
                            "classifier_id": 2,
                            "value": "b2",
                            "description": "b2"
                        },
                        {
                            "id": 3,
                            "classifier_id": 3,
                            "value": "c1",
                            "description": "c1"
                        }
                    ],
                    "merch_volume_to_biomass": {
                        'db_path': './cbm_defaults.db',
                        'merch_volume_curves': [
                            {
                                'classifier_set': {
                                    'type': 'name', 'values': ['a1','b2','c1']
                                },
                                'components': [
                                    {
                                    'species_id': 1,
                                    'age_volume_pairs': [(age0, vol0),
                                                         (age1, vol0),
                                                         (ageN, volN)]
                                    },
                                    {
                                    'species_id': 2,
                                    'age_volume_pairs': [(age0, vol0),
                                                         (age1, vol0),
                                                         (ageN, volN)]
                                    }
                                ]
                            }
                        ]
                    }
                }

        Raises:
            AssertionError: [description]
            RuntimeError: [description]
        """
        if not self.handle:
            raise AssertionError("dll not initialized")

        p_config = ctypes.c_char_p(config.encode("UTF-8"))

        self._dll.LibCBM_Initialize_CBM(ctypes.byref(self.err), self.handle,
                                        p_config)

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

    def AdvanceStandState(self, classifiers, spatial_units, disturbance_types,
                          transition_rule_ids, last_disturbance_type,
                          time_since_last_disturbance,
                          time_since_land_class_change, growth_enabled,
                          enabled, land_class, regeneration_delay, age):
        if not self.handle:
            raise AssertionError("dll not initialized")
        n = classifiers.shape[0]
        classifiersMat = LibCBM_Matrix_Int(classifiers)

        self._dll.LibCBM_AdvanceStandState(
            ctypes.byref(self.err), self.handle, n, classifiersMat,
            spatial_units, disturbance_types, transition_rule_ids,
            last_disturbance_type, time_since_last_disturbance,
            time_since_land_class_change, growth_enabled, enabled,
            land_class, regeneration_delay, age)

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

    def EndStep(self, age, regeneration_delay, enabled):
        if not self.handle:
            raise AssertionError("dll not initialized")
        n = age.shape[0]
        self._dll.LibCBM_EndStep(
            ctypes.byref(self.err), self.handle, n, age, regeneration_delay,
            enabled)

    def InitializeLandState(self, last_pass_disturbance, delay, initial_age,
                            spatial_units, afforestation_pre_type_id, pools,
                            last_disturbance_type, time_since_last_disturbance,
                            time_since_land_class_change, growth_enabled,
                            enabled, land_class, age):

        if not self.handle:
            raise AssertionError("dll not initialized")

        n = last_pass_disturbance.shape[0]
        poolMat = LibCBM_Matrix(pools)
        self._dll.LibCBM_InitializeLandState(
            ctypes.byref(self.err), self.handle, n, last_pass_disturbance,
            delay, initial_age, spatial_units, afforestation_pre_type_id,
            poolMat, last_disturbance_type, time_since_last_disturbance,
            time_since_land_class_change, growth_enabled, enabled, land_class,
            age)

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

    def AdvanceSpinupState(self, spatial_units, returnInterval, minRotations,
                           maxRotations, finalAge, delay, slowPools,
                           historical_disturbance, last_pass_disturbance,
                           afforestation_pre_type_id, state, disturbance_types,
                           rotation, step, lastRotationSlowC, enabled):
        if not self.handle:
            raise AssertionError("dll not initialized")
        n = spatial_units.shape[0]

        n_finished = self._dll.LibCBM_AdvanceSpinupState(
            ctypes.byref(self.err), self.handle, n,
            getNullableNdarray(spatial_units, type=ctypes.c_int),
            getNullableNdarray(returnInterval, type=ctypes.c_int),
            getNullableNdarray(minRotations, type=ctypes.c_int),
            getNullableNdarray(maxRotations, type=ctypes.c_int),
            finalAge, delay, slowPools, historical_disturbance,
            last_pass_disturbance, afforestation_pre_type_id, state,
            disturbance_types, rotation, step, lastRotationSlowC, enabled)

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

        return n_finished

    def EndSpinupStep(self, state, pools, disturbance_types, age, slowPools,
                      growth_enabled):
        if not self.handle:
            raise AssertionError("dll not initialized")
        n = age.shape[0]
        poolMat = LibCBM_Matrix(pools)
        self._dll.LibCBM_EndSpinupStep(
            ctypes.byref(self.err), self.handle, n, state, poolMat,
            disturbance_types, age, slowPools, growth_enabled)

    def GetMerchVolumeGrowthOps(self, growth_op, classifiers, pools, ages,
                                spatial_units, last_dist_type,
                                time_since_last_dist, growth_multipliers,
                                growth_enabled):
        if not self.handle:
            raise AssertionError("dll not initialized")
        n = pools.shape[0]
        poolMat = LibCBM_Matrix(pools)
        classifiersMat = LibCBM_Matrix_Int(classifiers)
        opIds = (ctypes.c_size_t * (1))(*[growth_op])
        self._dll.LibCBM_GetMerchVolumeGrowthOps(
            ctypes.byref(self.err), self.handle, opIds, n, classifiersMat,
            poolMat, ages, spatial_units,
            getNullableNdarray(last_dist_type, type=ctypes.c_int),
            getNullableNdarray(time_since_last_dist, type=ctypes.c_int),
            getNullableNdarray(growth_multipliers, type=ctypes.c_double),
            getNullableNdarray(growth_enabled, type=ctypes.c_int))

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

    def GetTurnoverOps(self, biomass_turnover_op, snag_turnover_op,
                       spatial_units):
        if not self.handle:
            raise AssertionError("dll not initialized")

        n = spatial_units.shape[0]
        opIds = (ctypes.c_size_t * (2))(
            *[biomass_turnover_op, snag_turnover_op])

        self._dll.LibCBM_GetTurnoverOps(
            ctypes.byref(self.err), self.handle, opIds, n, spatial_units)

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

    def GetDecayOps(self, dom_decay_op, slow_decay_op, slow_mixing_op,
                    spatial_units, historic_mean_annual_temp=False,
                    mean_annual_temp=None):
        if not self.handle:
            raise AssertionError("dll not initialized")
        n = spatial_units.shape[0]
        opIds = (ctypes.c_size_t * (3))(
            *[dom_decay_op, slow_decay_op, slow_mixing_op])
        self._dll.LibCBM_GetDecayOps(
            ctypes.byref(self.err), self.handle, opIds, n,
            getNullableNdarray(spatial_units, ctypes.c_int),
            historic_mean_annual_temp, getNullableNdarray(mean_annual_temp))
        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

    def GetDisturbanceOps(self, disturbance_op, spatial_units,
                          disturbance_type_ids):
        if not self.handle:
            raise AssertionError("dll not initialized")
        n = spatial_units.shape[0]
        opIds = (ctypes.c_size_t * (1))(*[disturbance_op])

        self._dll.LibCBM_GetDisturbanceOps(
            ctypes.byref(self.err), self.handle, opIds, n, spatial_units,
            disturbance_type_ids)

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())
