import ctypes, logging, sqlite3, os, numpy as np
from numpy.ctypeslib import ndpointer

class LibCBM_SpinupState:
    HistoricalRotation = 0,
    HistoricalDisturbance = 1,
    LastPassDisturbance = 2,
    GrowToFinalAge = 3,
    Delay = 4,
    Done = 5
    @staticmethod
    def getName(x):
        if x == 0: return "HistoricalRotation" 
        elif x == 1: return "HistoricalDisturbance"
        elif x == 2: return "LastPassDisturbance"
        elif x == 3: return "GrowToFinalAge"
        elif x == 4: return "Delay"
        elif x == 5: return "Done"
        else: raise ValueError("invalid Spinup state code")

class LibCBM_Matrix(ctypes.Structure):

    _fields_ = [('rows', ctypes.c_ssize_t),
                ('cols', ctypes.c_ssize_t),
                ('values', ctypes.POINTER(ctypes.c_double))]

    def __init__(self, matrix):
        if len(matrix.shape) == 1 and matrix.shape[0]==1:
            self.rows = 1
            self.cols = 1
        elif len(matrix.shape)==2:
            self.rows = matrix.shape[0]
            self.cols = matrix.shape[1]
        else:
            raise ValueError("matrix must have either 2 dimensions or be a single cell matrix")
        if not matrix.flags["C_CONTIGUOUS"] or not matrix.dtype == np.double:
            raise ValueError("matrix must be c contiguous and of type np.double")
        self.values = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

class LibCBM_Matrix_Int(ctypes.Structure):

    _fields_ = [('rows', ctypes.c_ssize_t),
                ('cols', ctypes.c_ssize_t),
                ('values', ctypes.POINTER(ctypes.c_int))]

    def __init__(self, matrix):
        if len(matrix.shape) == 1 and matrix.shape[0]==1:
            self.rows = 1
            self.cols = 1
        elif len(matrix.shape)==2:
            self.rows = matrix.shape[0]
            self.cols = matrix.shape[1]
        else:
            raise ValueError("matrix must have either 2 dimensions or be a single cell matrix")
        if not matrix.flags["C_CONTIGUOUS"] or not matrix.dtype == np.int32:
            raise ValueError("matrix must be c contiguous and of type np.int32")
        self.values = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

class LibCBM_Error(ctypes.Structure):
    _fields_ = [("Error", ctypes.c_int),
                ("Message", ctypes.ARRAY(ctypes.c_byte, 1000))]

    def __init__(self):
        setattr(self, "Error", 0)
        setattr(self, "Message", ctypes.ARRAY(ctypes.c_byte, 1000)())

    def getError(self):
        code = getattr(self, "Error")
        return code

    def getErrorMessage(self):
        msg = ctypes.cast(getattr(self, "Message"), ctypes.c_char_p).value
        return msg

def getNullableNdarray(a, type=ctypes.c_double):
    if a is None:
        return None
    else:
        result = np.ascontiguousarray(a).ctypes.data_as(ctypes.POINTER(type))
        return result

class LibCBMWrapper(object):
    def __init__(self, dllpath):
        self.handle = False
        #necessary because supporting libraries are in the same dir as the main one
        #this needs to be fixed (will likely switch to static library)
        dlldir = os.path.dirname(dllpath)
        cwd = os.getcwd()
        os.chdir(os.path.dirname(dllpath))
        self._dll = ctypes.CDLL(dllpath)
        os.chdir(cwd)
        self.err = LibCBM_Error();

        self._dll.LibCBM_Free.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p # handle pointer
        )

        self._dll.LibCBM_Initialize.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_char_p # config json string
        )

        self._dll.LibCBM_Allocate_Op.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.c_size_t #n ops
        )

        self._dll.LibCBM_SetOp.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.c_size_t, #op_id
            ctypes.POINTER(LibCBM_Matrix),#matrices
            ctypes.c_size_t, #n_matrices
            ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"), #matrix_index
            ctypes.c_size_t #n_matrix_index
        )

        self._dll.LibCBM_ComputePools.argtypes = (
                ctypes.POINTER(LibCBM_Error), # error struct
                ctypes.c_void_p, #handle
                ctypes.POINTER(ctypes.c_size_t), #op ids
                ctypes.c_size_t, #number of op ids
                LibCBM_Matrix, #pools
                ctypes.POINTER(ctypes.c_int) #enabled
            )

        self._dll.LibCBM_ComputeFlux.argtypes = (
                ctypes.POINTER(LibCBM_Error), # error struct
                ctypes.c_void_p, #handle
                ctypes.POINTER(ctypes.c_size_t), #op ids
                ctypes.POINTER(ctypes.c_size_t), #op process ids
                ctypes.c_size_t, #number of ops
                LibCBM_Matrix, # pools (nstands by npools)
                LibCBM_Matrix, # flux (nstands by nfluxIndicators)
                ctypes.POINTER(ctypes.c_int) #enabled
            )

        self._dll.LibCBM_Initialize_CBM.argtypes = (
                ctypes.POINTER(LibCBM_Error), # error struct
                ctypes.c_void_p, #handle
                ctypes.c_char_p # config json string
            )

        self._dll.LibCBM_AdvanceStandState.argtypes = (
                ctypes.POINTER(LibCBM_Error), # error struct
                ctypes.c_void_p, #handle
                ctypes.c_size_t, #n stands
                LibCBM_Matrix_Int, #classifiers
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # disturbance_types (length n)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # transition_rule_ids (length n)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # last_disturbance_type (length n) (return value)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # time_since_last_disturbance (length n) (return value)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # time_since_land_class_change (length n) (return value)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # growth_enabled (length n) (return value)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # enabled (length n) (return value)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # land_class (length n) (return value)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # regeneration_delay (length n) (return value)
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # age (length n) (return value)
            )

        self._dll.LibCBM_EndStep.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.c_size_t, #n stands
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),#age
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS") #regeneration_delay
            )

        self._dll.LibCBM_InitializeLandState.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.c_size_t, #n stands
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #last_pass_disturbance (length n)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #delay (length n)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #initial_age (length n)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #spatial unit id (length n)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #afforestation pre type id (length n)
            LibCBM_Matrix, # pools
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #last_disturbance_type (length n) (return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #time_since_last_disturbance (length n) (return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #time_since_land_class_change (length n) (return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #growth_enabled (length n) (return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #enabled (length n) (return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #land_class (length n) (return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")  #age (length n) (return value)
        )

        self._dll.LibCBM_AdvanceSpinupState.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.c_size_t, #n stands
            ctypes.POINTER(ctypes.c_int), #spatial unit id not using ndpointer becuase we are allowing null(length n)
            ctypes.POINTER(ctypes.c_int), #return interval not using ndpointer becuase we are allowing null(length n)
            ctypes.POINTER(ctypes.c_int), #minRotations not using ndpointer becuase we are allowing null(length n)
            ctypes.POINTER(ctypes.c_int), #maxRotations not using ndpointer becuase we are allowing null(length n)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #final age (length n)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #delay (length n)
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), #slowpools (length n)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # historical disturbance type (length n)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # last pass disturbance type (length n)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #afforestation pre type id (length n)
            ndpointer(ctypes.c_uint, flags="C_CONTIGUOUS"), #spinup state code (length n) (return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #disturbance type  (length n)(return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #Rotation num (length n)(return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #simulation step (length n)(return value)
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), #last rotation slow (length n)(return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS") #enabled
        )

        self._dll.LibCBM_EndSpinupStep.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.c_size_t, #n stands
            ndpointer(ctypes.c_uint, flags="C_CONTIGUOUS"), #spinup state code (length n)
            LibCBM_Matrix, # pools
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #age (length n)(return value)
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS") #sum of slow pools (length n) (return value)
            )

        self._dll.LibCBM_GetMerchVolumeGrowthOps.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.ARRAY(ctypes.c_size_t, 1), #op_ids 
            ctypes.c_size_t, #n stands
            LibCBM_Matrix_Int, #classifiers
            LibCBM_Matrix, #pools
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #stand ages (length n)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #spatial unit id (length n)
            ctypes.POINTER(ctypes.c_int), # (nullable) last disturbance type (length n)
            ctypes.POINTER(ctypes.c_int), # (nullable) time since last disturbance (length n)
            ctypes.POINTER(ctypes.c_double), # (nullable) growth multiplier (length n)
            ctypes.POINTER(ctypes.c_int) # (nullable) growth enabled (length n)
            )

        self._dll.LibCBM_GetTurnoverOps.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.ARRAY(ctypes.c_size_t,2), #op_ids
            ctypes.c_size_t, #n stands
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")) #spatial unit id (length n)

        self._dll.LibCBM_GetDecayOps.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.ARRAY(ctypes.c_size_t,3), #op_ids
            ctypes.c_size_t, #n stands
            ctypes.POINTER(ctypes.c_int), #spatial unit id  not using ndpointer becuase we are allowing null (length n)
            ctypes.c_bool, #use historic mean annual temp
            ctypes.POINTER(ctypes.c_double) #mean annual temp, not using ndpointer becuase we are allowing null
            )

        self._dll.LibCBM_GetDisturbanceOps.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.ARRAY(ctypes.c_size_t,1), #op_id
            ctypes.c_size_t, #n stands
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #spatial unit id (length n)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #disturbance type ids (length n)
            )


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle:
            err = LibCBM_Error()
            self._dll.LibCBM_Free(ctypes.byref(err), self.handle)
            if err.Error != 0:
                raise RuntimeError(err.getErrorMessage())

    def Initialize(self, config):

        p_config = ctypes.c_char_p(config.encode("UTF-8"));

        self.handle = self._dll.LibCBM_Initialize(
            ctypes.byref(self.err), #error struct
            p_config
            )

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())


    def AllocateOp(self, n):
        if not self.handle:
           raise AssertionError("dll not initialized")

        op_id = self._dll.LibCBM_Allocate_Op(
            ctypes.byref(self.err),
            self.handle, n)

        if self.err.Error != 0:
           raise RuntimeError(self.err.getErrorMessage())

        return op_id

    def SetOp(self, op_id, matrices, matrix_index):
        if not self.handle:
           raise AssertionError("dll not initialized")
        matrices_array = (LibCBM_Matrix * len(matrices))()
        for i,x in enumerate(matrices):
            matrices_array[i] = LibCBM_Matrix(x)
        matrices_p =  ctypes.cast(matrices_array, ctypes.POINTER(LibCBM_Matrix))
        self._dll.LibCBM_SetOp(ctypes.byref(self.err), self.handle, op_id,
            matrices_p, len(matrices), matrix_index, matrix_index.shape[0])

    def ComputePools(self, ops, pools, enabled=None):

       if not self.handle:
           raise AssertionError("dll not initialized")

       n_ops = len(ops)
       poolMat = LibCBM_Matrix(pools)
       ops_p = ctypes.cast((ctypes.c_size_t*n_ops)(*ops), ctypes.POINTER(ctypes.c_size_t))

       self._dll.LibCBM_ComputePools(ctypes.byref(self.err), self.handle, ops_p,
            n_ops, poolMat, getNullableNdarray(enabled, type = ctypes.c_int))

       if self.err.Error != 0:
           raise RuntimeError(self.err.getErrorMessage())


    def ComputeFlux(self, ops, op_processes, pools, flux, enabled=None):
        if not self.handle:
           raise AssertionError("dll not initialized")

        n_ops = len(ops)
        if len(op_processes) != n_ops:
            raise ValueError("ops and op_processes must be of equal length")
        poolMat = LibCBM_Matrix(pools)
        fluxMat = LibCBM_Matrix(flux)

        ops_p = ctypes.cast((ctypes.c_size_t*n_ops)(*ops), ctypes.POINTER(ctypes.c_size_t))
        op_process_p = ctypes.cast((ctypes.c_size_t*n_ops)(*op_processes), ctypes.POINTER(ctypes.c_size_t))

        self._dll.LibCBM_ComputeFlux(ctypes.byref(self.err), self.handle,
            ops_p, op_process_p, n_ops, poolMat, fluxMat,
            getNullableNdarray(enabled, type = ctypes.c_int))

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())


    def InitializeCBM(self, config):
        if not self.handle:
           raise AssertionError("dll not initialized")

        p_config = ctypes.c_char_p(config.encode("UTF-8"));

        self._dll.LibCBM_Initialize_CBM(ctypes.byref(self.err), self.handle,
                                        p_config)

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())


    def AdvanceStandState(self, classifiers, disturbance_types, transition_rule_ids,
                          last_disturbance_type, time_since_last_disturbance,
                          time_since_land_class_change, growth_enabled, enabled,
                          land_class, regeneration_delay, age):
       if not self.handle:
           raise AssertionError("dll not initialized")
       n = classifiers.shape[0]
       classifiersMat = LibCBM_Matrix_Int(classifiers)

       self._dll.LibCBM_AdvanceStandState(
            ctypes.byref(self.err), self.handle, n, classifiersMat,
            disturbance_types, transition_rule_ids, last_disturbance_type,
            time_since_last_disturbance, time_since_land_class_change,
            growth_enabled, enabled, land_class, regeneration_delay, age)

       if self.err.Error != 0:
           raise RuntimeError(self.err.getErrorMessage())


    def EndStep(self, age, regeneration_delay):
        if not self.handle:
            raise AssertionError("dll not initialized")
        n = age.shape[0]
        self._dll.LibCBM_EndStep(ctypes.byref(self.err), self.handle, n, age,
            regeneration_delay)


    def InitializeLandState(self, last_pass_disturbance, delay, initial_age,
        spatial_units, afforestation_pre_type_id, pools, last_disturbance_type,
        time_since_last_disturbance, time_since_land_class_change,
        growth_enabled, enabled, land_class, age):

        if not self.handle:
            raise AssertionError("dll not initialized")
        n = last_pass_disturbance.shape[0]
        self._dll.LibCBM_InitializeLandState(ctypes.byref(self.err),
            self.handle, n, last_pass_disturbance, delay, initial_age,
            spatial_units, afforestation_pre_type_id, pools,
            last_disturbance_type, time_since_last_disturbance,
            time_since_land_class_change, growth_enabled, enabled, land_class, age)

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
            getNullableNdarray(spatial_units, type = ctypes.c_int),
            getNullableNdarray(returnInterval, type = ctypes.c_int),
            getNullableNdarray(minRotations, type = ctypes.c_int),
            getNullableNdarray(maxRotations, type = ctypes.c_int),
            finalAge, delay, slowPools,historical_disturbance,
            last_pass_disturbance, afforestation_pre_type_id, state,
            disturbance_types, rotation, step, lastRotationSlowC, enabled)

       if self.err.Error != 0:
           raise RuntimeError(self.err.getErrorMessage())

       return n_finished


    def EndSpinupStep(self, state, pools, age, slowPools):
        if not self.handle:
            raise AssertionError("dll not initialized")
        n = age.shape[0]
        poolMat = LibCBM_Matrix(pools)
        self._dll.LibCBM_EndSpinupStep(ctypes.byref(self.err), self.handle, n,
            state, poolMat, age, slowPools)


    def GetMerchVolumeGrowthOps(self, growth_op, 
        classifiers, pools, ages, spatial_units, last_dist_type,
        time_since_last_dist, growth_multipliers, growth_enabled):

        if not self.handle:
            raise AssertionError("dll not initialized")
        n = pools.shape[0]
        poolMat = LibCBM_Matrix(pools)
        classifiersMat = LibCBM_Matrix_Int(classifiers)
        opIds = (ctypes.c_size_t * (1))(*[growth_op])
        self._dll.LibCBM_GetMerchVolumeGrowthOps(ctypes.byref(self.err),
            self.handle, opIds, n, classifiersMat, poolMat, ages,
            spatial_units,
            getNullableNdarray(last_dist_type, type = ctypes.c_int),
            getNullableNdarray(time_since_last_dist, type = ctypes.c_int),
            getNullableNdarray(growth_multipliers, type=ctypes.c_double),
            getNullableNdarray(growth_enabled, type = ctypes.c_int))

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

    def GetTurnoverOps(self, biomass_turnover_op, snag_turnover_op,
                       spatial_units):
        if not self.handle:
           raise AssertionError("dll not initialized")

        n = spatial_units.shape[0]
        opIds = (ctypes.c_size_t * (2))(*[biomass_turnover_op,snag_turnover_op])

        self._dll.LibCBM_GetTurnoverOps(ctypes.byref(self.err), self.handle,
           opIds, n, spatial_units)

        if self.err.Error != 0:
           raise RuntimeError(self.err.getErrorMessage())

    def GetDecayOps(self, dom_decay_op, slow_decay_op, slow_mixing_op,
                    spatial_units, historic_mean_annual_temp = False,
                    mean_annual_temp=None):
        if not self.handle:
           raise AssertionError("dll not initialized")
        n = spatial_units.shape[0]
        opIds = (ctypes.c_size_t * (3))(
            *[dom_decay_op, slow_decay_op, slow_mixing_op])
        self._dll.LibCBM_GetDecayOps(ctypes.byref(self.err), self.handle,
            opIds, n, getNullableNdarray(spatial_units, ctypes.c_int),
            historic_mean_annual_temp, getNullableNdarray( mean_annual_temp))
        if self.err.Error != 0:
           raise RuntimeError(self.err.getErrorMessage())


    def GetDisturbanceOps(self, disturbance_op, spatial_units, disturbance_type_ids):
        if not self.handle:
           raise AssertionError("dll not initialized")
        n = spatial_units.shape[0]
        opIds = (ctypes.c_size_t * (1))(*[disturbance_op])

        self._dll.LibCBM_GetDisturbanceOps(ctypes.byref(self.err), self.handle,
            opIds, n, spatial_units, disturbance_type_ids)

        if self.err.Error != 0:
           raise RuntimeError(self.err.getErrorMessage())
