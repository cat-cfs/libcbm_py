import ctypes, logging, sqlite3, numpy as np
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

def isNumpyArray(value):
    if isinstance(value, np.ndarray):
        return True
    return False

class LibCBM_Matrix(ctypes.Structure):

    _fields_ = [('rows', ctypes.c_ssize_t),
                ('cols', ctypes.c_ssize_t),
                ('values', ctypes.POINTER(ctypes.c_double))]

    def _init_(self, nrows, ncols):
        self.rows = nrows
        self.cols = ncols
        init = [0.0 for x in range(0,nrows*ncols)]
        self.values = (ctypes.c_double * (nrows * ncols))(*init)

    def _init_np(self, np_matrix):
        self.rows = np_matrix.shape[0]
        self.cols = np_matrix.shape[1]
        self.values = np_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def __init__(self, matrix):
        if isNumpyArray(matrix):
            self._init_np(matrix)
        else:
            self._init_(matrix[0], matrix[1])

class LibCBM_Matrix_Int(ctypes.Structure):

    _fields_ = [('rows', ctypes.c_ssize_t),
                ('cols', ctypes.c_ssize_t),
                ('values', ctypes.POINTER(ctypes.c_int))]

    def _init_(self, nrows, ncols):
        """
        use this method for matrices that may be modified by the Dll.
        The initial value of all data is 0, and the total size is ncols*nrows
        @param nrows the number of rows in the matrix
        @param ncols the number of columns in the matrix
        """
        self.rows = nrows
        self.cols = ncols
        init = [0 for x in range(0,nrows*ncols)]
        self.values = (ctypes.c_int * (nrows * ncols))(*init)

    def _init_np(self, np_matrix):
        """
        use this method for matrices that are immutable input data for the
        Sawtooth Dll.
        @param np_matrix a numpy matrix with shape order 2 and integer data 
        type
        """
        self.rows = np_matrix.shape[0]
        self.cols = np_matrix.shape[1]
        if np_matrix.dtype != np.integer:
            raise ValueError("specified value {0} is not an integer".format(np_matrix.dtype))
        self.values = np_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    def __init__(self, matrix):
        if isNumpyArray(matrix):
            self._init_np(matrix)
        else:
            self._init_(matrix[0], matrix[1])

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

class LibCBM_Classifier(ctypes.Structure):
    _fields_ = [("id", ctypes.c_size_t),
                ("name", ctypes.c_char_p)]
    
    def __init__(self, id, name):
        setattr(self,"id",id)
        setattr(self,"name", ctypes.c_char_p(name))

class LibCBM_ClassifierValue(ctypes.Structure):
    _fields_ = [("id", ctypes.c_size_t),
                ("classifier_id", ctypes.c_size_t),
                ("name", ctypes.c_char_p),
                ("description", ctypes.c_char_p)]

    def __init__(self, id, classifier_id, name, description):
        setattr(self,"id",id)
        setattr(self,"classifier_id", classifier_id)
        setattr(self,"name", ctypes.c_char_p(name))
        setattr(self,"description", ctypes.c_char_p(description))

class LibCBM_MerchVolumeComponent(ctypes.Structure):
    _fields_ = [("species_id", ctypes.c_int),
                ("num_values", ctypes.c_size_t),
                ("age", ctypes.POINTER(ctypes.c_int)),
                ("volume", ctypes.POINTER(ctypes.c_double))]

    def __init__(self, species_id, ages, volumes):
        setattr(self, "species_id", species_id)
        setattr(self, "num_values", len(ages))
        setattr(self, "age", (ctypes.c_int * len(ages))(*ages))
        setattr(self, "volume", (ctypes.c_double * len(volumes))(*volumes))

class LibCBM_MerchVolumeCurve(ctypes.Structure):
    _fields_ = [("classifierValueIds", ctypes.POINTER(ctypes.c_size_t)),
                ("nClassifierValueIds", ctypes.c_size_t),
                ("SoftwoodComponent", ctypes.POINTER(LibCBM_MerchVolumeComponent)),
                ("HardwoodComponent", ctypes.POINTER(LibCBM_MerchVolumeComponent))]

    def __init__(self, classifierValueIds, softwoodComponent, hardwoodComponent):
        setattr(self, "classifierValueIds", 
                (ctypes.c_size_t * len(classifierValueIds))
                (*classifierValueIds))
        setattr(self, "nClassifierValueIds", len(classifierValueIds))
        setattr(self, "SoftwoodComponent", ctypes.pointer(softwoodComponent))
        setattr(self, "HardwoodComponent", ctypes.pointer(hardwoodComponent))

class LibCBMWrapper(object):
    def __init__(self, dllpath):
        self.NFluxIndicators = 0
        self.PoolCount = 27
        self.NProcesses = 9
        self.handle = False
        self._dll = ctypes.CDLL(dllpath)
        self.err = LibCBM_Error();

        self._dll.LibCBM_Free.argtypes = (
            ctypes.POINTER(LibCBM_Error),
            ctypes.c_void_p)

        self._dll.LibCBM_Initialize.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_char_p, # dbpath
            ctypes.c_size_t, # random seed
            ctypes.POINTER(LibCBM_Classifier), # classifiers
            ctypes.c_size_t, # number of classifiers
            ctypes.POINTER(LibCBM_ClassifierValue), # classifier values
            ctypes.c_size_t, # number of classifier values
            ctypes.POINTER(LibCBM_MerchVolumeCurve), # merch volume curves
            ctypes.c_size_t #number of merch vol curves
        )

        self._dll.LibCBM_AdvanceSpinupState.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.c_size_t, #n stands
            ctypes.POINTER(ctypes.c_int), #return interval (length n)
            ctypes.POINTER(ctypes.c_int), #minRotations (length n)
            ctypes.POINTER(ctypes.c_int), #maxRotations (length n)
            ctypes.POINTER(ctypes.c_int), #final age (length n)
            ctypes.POINTER(ctypes.c_int), #delay (length n)
            ctypes.POINTER(ctypes.c_double), #slowpools (length n)
            ndpointer(ctypes.c_uint, flags="C_CONTIGUOUS"), #spinup state code (length n) (return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #Rotation num (length n)(return value)
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), #simulation step (length n)(return value)
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), #last rotation slow (length n)(return value)
            )

        self._dll.LibCBM_GetMerchVolumeGrowthAndDeclineOps.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.ARRAY(ctypes.c_size_t,2), #op_ids (returned value)
            ctypes.c_size_t, #n stands
            LibCBM_Matrix_Int, #classifiers
            LibCBM_Matrix, #pools
            ctypes.POINTER(ctypes.c_int), #stand ages (length n)
            ctypes.POINTER(ctypes.c_int), #spatial unit id (length n)
            ctypes.POINTER(ctypes.c_int), # last disturbance type (length n)
            ctypes.POINTER(ctypes.c_int), # time since last disturbance
            ctypes.POINTER(ctypes.c_double) #growth multiplier
            )

        self._dll.LibCBM_GetTurnoverOps.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.ARRAY(ctypes.c_size_t,2), #op_ids (returned value))
            ctypes.c_size_t, #n stands
            ctypes.POINTER(ctypes.c_int)) #spatial unit id (length n)

        self._dll.LibCBM_GetDecayOps.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.ARRAY(ctypes.c_size_t,3), #op_ids (returned value))
            ctypes.c_size_t, #n stands
            ctypes.POINTER(ctypes.c_int), #spatial unit id (length n)
            ctypes.POINTER(ctypes.c_double), #mean annual temp
            ctypes.c_bool #use mean annual temp 
            )

        self._dll.LibCBM_GetDisturbanceOps.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.ARRAY(ctypes.c_size_t,1), #op_ids (returned value))
            ctypes.c_size_t, #n stands
            ctypes.POINTER(ctypes.c_int), #spatial unit id (length n)
            ctypes.POINTER(ctypes.c_int) #disturbacne type id
            )

        #self._dll.LibCBM_Spinup.argtypes = (
        #    ctypes.POINTER(LibCBM_Error), # error struct
        #    ctypes.c_void_p, #handle
        #    ctypes.POINTER(ctypes.c_double), #pool result
        #    ctypes.POINTER(ctypes.c_size_t), #classifier set
        #    ctypes.c_size_t, #n classifiers
        #    ctypes.c_int, # spatial unit id
        #    ctypes.c_int, #age
        #    ctypes.c_int, #delay
        #    ctypes.c_double, #mean annual temp
        #    ctypes.c_bool, #use default temp
        #    ctypes.c_int, #historical disturbance type
        #    ctypes.c_int, #last pass disturbance type
        #    ctypes.c_bool, #use random return interval (uncertainty analysis)
        #    ctypes.c_bool #use caching
        #)

        #self._dll.LibCBM_Step.argtypes = (
        #    ctypes.POINTER(LibCBM_Error), # error struct
        #    ctypes.c_void_p, #handle
        #    ctypes.POINTER(ctypes.c_double), #pools t0
        #    ctypes.POINTER(ctypes.c_double), #pools t1
        #    ctypes.POINTER(ctypes.c_size_t), #classifier set
        #    ctypes.c_size_t, #n classifiers
        #    ctypes.c_int, #age
        #    ctypes.c_int, #spatial_unit_id
        #    ctypes.c_int, #last disturbance type
        #    ctypes.c_int, #time since last disturbance
        #    ctypes.c_double, #mean annual temp
        #    ctypes.c_bool, # use default temp
        #    ctypes.c_int, #disturbance type id
        #    ctypes.POINTER(ctypes.c_double), #flux indicator values
        #    ctypes.POINTER(LibCBM_CoordinateMatrix), #raw pool flows
        #    )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle:
            err = LibCBM_Error()
            self._dll.LibCBM_Free(ctypes.byref(err), self.handle)
            if err.Error != 0:
                raise RuntimeError(err.getErrorMessage())

    def Initialize(self, dbpath, random_seed,
                   classifiers, classifierValues,
                   merchVolumeCurves):

        _classifiers = [LibCBM_Classifier(x["id"], x["name"])
                       for x in classifiers]

        _classifierValues = [
            LibCBM_ClassifierValue(
                x["id"],
                x["classifier_id"],
                x["name"],
                x["description"])
            for x in classifierValues
        ]

        _merchVolumeCurves = [
            LibCBM_MerchVolumeCurve(
                x["classifier_value_ids"],
                LibCBM_MerchVolumeComponent(
                    x["sw_component"]["species_id"],
                    x["sw_component"]["ages"],
                    x["sw_component"]["volumes"]),
                LibCBM_MerchVolumeComponent(
                    x["hw_component"]["species_id"],
                    x["hw_component"]["ages"],
                    x["hw_component"]["volumes"])
                )
           for x in merchVolumeCurves
         ]

        _classifiers_p = (LibCBM_Classifier*len(_classifiers))(*_classifiers)
        _classifierValue_p = (LibCBM_ClassifierValue*len(_classifierValues))(*_classifierValues)
        _merchVolumeCurves_p = (LibCBM_MerchVolumeCurve*len(_merchVolumeCurves))(*_merchVolumeCurves)

        self.handle = self._dll.LibCBM_Initialize(
            ctypes.byref(self.err), #error struct
            dbpath, #path to cbm defaults database
            1, #random seed
            _classifiers_p, len(_classifiers),
            _classifierValue_p, len(_classifierValues),
            _merchVolumeCurves_p, len(_merchVolumeCurves)
            )

        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

    def AdvanceSpinupState(self, returnInterval, minRotations, maxRotations,
                           finalAge, delay, slowPools, state, rotation, step,
                           lastRotationSlowC):
       if not self.handle:
           raise AssertionError("dll not initialized")
       n = returnInterval.shape[0]

       self._dll.LibCBM_AdvanceSpinupState(
            ctypes.byref(self.err),
            self.handle,
            n,
            returnInterval.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            minRotations.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            maxRotations.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            finalAge.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            delay.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            slowPools.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            state,
            rotation,
            step,
            lastRotationSlowC)
        

       if self.err.Error != 0:
           raise RuntimeError(self.err.getErrorMessage())

    def GetMerchVolumeGrowthAndDeclineOps(self,
        classifiers, pools, ages, spatial_units, last_dist_type,
        time_since_last_dist, growth_multipliers):

       if not self.handle:
           raise AssertionError("dll not initialized")
       n = pools.shape[0]
       poolMat = LibCBM_Matrix(pools)
       classifiersMat = LibCBM_Matrix_Int(classifiers)
       opIds = (ctypes.c_size_t * (2))(*[0,0])
       self._dll.LibCBM_GetMerchVolumeGrowthAndDeclineOps(
           ctypes.byref(self.err),
           self.handle,
           opIds,
           n,
           classifiersMat, poolMat,
           ages.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
           spatial_units.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
           last_dist_type.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
           time_since_last_dist.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
           growth_multipliers.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

       if self.err.Error != 0:
           raise RuntimeError(self.err.getErrorMessage())

       return {
           opIds[0]: "Growth",
           opIds[1]: "OvermatureDecline"
           }

    def GetTurnoverOps(self, spatial_units):
        if not self.handle:
           raise AssertionError("dll not initialized")

        n = spatial_units.shape[0]
        opIds = (ctypes.c_size_t * (2))(*[0,0])

        self._dll.LibCBM_GetTurnoverOps(
           ctypes.byref(self.err),
           self.handle,
           opIds,
           n,
           spatial_units.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))

        if self.err.Error != 0:
           raise RuntimeError(self.err.getErrorMessage())

        return {
           opIds[0]: "BiomassTurnover",
           opIds[1]: "SnagTurnover"
           }

    def GetDecayOps(self, spatial_units, mean_annual_temp=None):
        if not self.handle:
           raise AssertionError("dll not initialized")
        n = spatial_units.shape[0]
        opIds = (ctypes.c_size_t * (3))(*[0,0,0])
        self._dll.LibCBM_GetDecayOps(
           ctypes.byref(self.err),
           self.handle,
           opIds,
           n,
           spatial_units.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
           None if mean_annual_temp is None else #null pointer if no mean annual temp specified
                mean_annual_temp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
           False if mean_annual_temp is None else True)

        if self.err.Error != 0:
           raise RuntimeError(self.err.getErrorMessage())

        return {
           opIds[0]: "DomDecay",
           opIds[1]: "SlowDecay",
           opIds[2]: "SlowMixing",
           }

    def GetDisturbanceOps(self, spatial_units, disturbance_types):
        if not self.handle:
           raise AssertionError("dll not initialized")
        n = spatial_units.shape[0]
        opIds = (ctypes.c_size_t * (1))(*[0])

        self._dll.LibCBM_GetDisturbanceOps(ctypes.byref(self.err),
           self.handle,
           opIds,
           n,
           spatial_units.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
           disturbance_types.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))

        if self.err.Error != 0:
           raise RuntimeError(self.err.getErrorMessage())
        return {
           opIds[0]: "Disturbance"
           }
    #def Spinup(self, classifierSet, spatial_unit_id, age, delay,
    #           historical_disturbance_type_id, last_pass_disturbance_type_id,
    #           use_cache=True, random_return_interval =False,
    #           mean_annual_temp=None):

    #    if not self.handle:
    #        raise AssertionError("dll not initialized")

    #    pools = np.ndarray((self.PoolCount,))
    #    pools_p = pools.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    #    classifiers_p = (ctypes.c_size_t * len(classifierSet))(*classifierSet)

    #    self._dll.LibCBM_Spinup(
    #        ctypes.byref(self.err), #error struct
    #        self.handle,
    #        pools_p,
    #        classifiers_p,
    #        len(classifierSet),
    #        spatial_unit_id,
    #        age,
    #        delay,
    #        0.0 if mean_annual_temp is None else mean_annual_temp,
    #        mean_annual_temp is None,
    #        historical_disturbance_type_id,
    #        last_pass_disturbance_type_id,
    #        random_return_interval,
    #        use_cache
    #        )

    #    if self.err.Error != 0:
    #        raise RuntimeError(self.err.getErrorMessage())

    #    return pools

    #def Step(self, pools, classifierSet, age, spatial_unit_id,
    #         lastDisturbanceType, timeSinceLastDisturbance,
    #         disturbance_type_id=0, mean_annual_temp=None,
    #         get_raw_flux=False):

    #    pools_p = pools.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    #    pools_t1 = np.ndarray((self.PoolCount,))
    #    pools_t1_p = pools_t1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    #    classifiers_p = (ctypes.c_size_t * len(classifierSet))(*classifierSet)

    #    raw_flux_p = None
    #    if get_raw_flux:
    #        raw_flux_p = (LibCBM_CoordinateMatrix * self.NProcesses) \
    #            (*[LibCBM_CoordinateMatrix(128) for x in xrange(self.NProcesses)])
    #    else:
    #        raw_flux_p = ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(LibCBM_CoordinateMatrix))

    #    fluxIndicators = np.ndarray((self.NFluxIndicators,))
    #    fluxIndicators_p = fluxIndicators.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    #    self._dll.LibCBM_Step(
    #        ctypes.byref(self.err), #error struct
    #        self.handle,
    #        pools_p,
    #        pools_t1_p,
    #        classifiers_p,
    #        len(classifierSet),
    #        age,
    #        spatial_unit_id,
    #        lastDisturbanceType,
    #        timeSinceLastDisturbance,
    #        0.0 if mean_annual_temp is None else mean_annual_temp,
    #        mean_annual_temp is None,
    #        disturbance_type_id,
    #        fluxIndicators_p,
    #        raw_flux_p)

    #    return {
    #        "Pools": pools_t1,
    #        "FluxIndicators": fluxIndicators,
    #        "Flows": raw_flux_p
    #        }

