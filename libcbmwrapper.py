import ctypes, logging, numpy as np
from numpy.ctypeslib import ndpointer

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

class LibCBM_CoordinateMatrix(ctypes.Structure):
    _fields_ = [("memsize", ctypes.c_size_t),
                ("count", ctypes.c_size_t),
                ("rows", ctypes.POINTER(ctypes.c_size_t)),
                ("cols", ctypes.POINTER(ctypes.c_size_t)),
                ("values", ctypes.POINTER(ctypes.c_double))]

    def __init__(self, size):
        setattr(self, "memsize", size)
        setattr(self, "count", 0)
        setattr(self, "rows", (ctypes.c_size_t * size))(*[0]*size)
        setattr(self, "cols", (ctypes.c_size_t * size))(*[0]*size)
        setattr(self, "values", (ctypes.c_double * size))(*[0]*size)

class LibCBM_FluxIndicator(ctypes.Structure):
    _fields_ = [
        ("includedProcesses", ctypes.POINTER(ctypes.c_size_t)),#included processes
        ("nIncludedProcesses", ctypes.c_size_t),#number of included processes
        ("eligibleSources", ctypes.POINTER(ctypes.c_size_t)),#included source pools
        ("nEligibleSources", ctypes.c_size_t),#number of included source pools
        ("eligibleSinks", ctypes.POINTER(ctypes.c_size_t)),#included sink pools
        ("nEligibleSinks", ctypes.c_size_t),#number of included sink pools
        ]

    def __init__(self, processes, sourcePools, sinkPools):
        setattr(self, "includedProcesses", (ctypes.c_size_t * len(processes)(processes)))
        setattr(self, "nIncludedProcesses", len(processes))
        setattr(self, "eligibleSources", (ctypes.c_size_t * len(sourcePools)(sourcePools)))
        setattr(self, "nEligibleSources", len(sourcePools))
        setattr(self, "eligibleSinks", (ctypes.c_size_t * len(sinkPools)(sinkPools)))
        setattr(self, "nEligibleSinks", len(sinkPools))

class LibCBMWrapper(object):
    def __init__(self, dllpath):
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

        self._dll.LibCBM_Spinup.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.POINTER(ctypes.c_double), #pool result
            ctypes.POINTER(ctypes.c_size_t), #classifier set
            ctypes.c_size_t, #n classifiers
            ctypes.c_int, # spatial unit id
            ctypes.c_int, #age
            ctypes.c_int, #delay
            ctypes.c_double, #mean annual temp
            ctypes.c_bool, #use default temp
            ctypes.c_int, #historical disturbance type
            ctypes.c_int, #last pass disturbance type
            ctypes.c_bool, #use random return interval (uncertainty analysis)
            ctypes.c_bool #use caching
        )

        self._dll.LibCBM_Step.argtypes = (
            ctypes.POINTER(LibCBM_Error), # error struct
            ctypes.c_void_p, #handle
            ctypes.POINTER(ctypes.c_double), #pools t0
            ctypes.POINTER(ctypes.c_double), #pools t1
            ctypes.POINTER(ctypes.c_size_t), #classifier set
            ctypes.c_size_t, #n classifiers
            ctypes.c_int, #age
            ctypes.c_int, #spatial_unit_id
            ctypes.c_int, #last disturbance type
            ctypes.c_int, #time since last disturbance
            ctypes.c_double, #mean annual temp
            ctypes.c_bool, # use default temp
            ctypes.c_int, #disturbance type id
            ctypes.POINTER(LibCBM_CoordinateMatrix), #pool flows
            ctypes.POINTER(ctypes.c_double) #flux indicator values
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle:
            err = LibCBM_Error()
            self.sawtoothDLL.LibCBM_Free(ctypes.byref(err), self.handle)
            if err.Error != 0:
                raise RuntimeError(err.getErrorMessage())

    def InitializeMerchVolumeComponent(self):
        pass

    def Initialize(self, dbpath, random_seed,
                   classifiers, classifierValues,
                   merchVolumeCurves, fluxIndicators):

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

        _fluxIndicators = [
            LibCBM_FluxIndicator(x["Processes"], x["SourcePools"], x["SinkPools"])

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

    def Spinup(self, ):
        if not self.handle:
            raise AssertionError("dll not initialized")

        pools = [0 for x in range(27)]
        pools_p = (ctypes.c_double * len(pools))(*pools)

        self._dll.LibCBM_Spinup(
            ctypes.byref(self.err), #error struct
            self.handle,
            pools_p,

            )
