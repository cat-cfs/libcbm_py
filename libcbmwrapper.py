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
        setattr(self,"name",name)

class LibCBM_ClassifierValue(ctypes.Structure):
    _fields_ = [("id", ctypes.c_size_t),
                ("classifier_id", ctypes.c_size_t),
                ("name", ctypes.c_char_p),
                ("description", ctypes.c_char_p)]

    def __init__(self, id, name):
        setattr(self,"id",id)
        setattr(self,"name",name)

class LibCBM_MerchVolumeComponent(ctypes.Structure):
    _fields_ = [("species_id", ctypes.c_int),
                ("num_values", ctypes.c_size_t),
                ("age", POINTER(ctypes.c_int)),
                ("volume", POINTER(ctypes.c_double))]

    def __init__(self, species_id, ageVolumePairs):
        pass

class LibCBM_MerchVolumeCurve(ctypes.Structure):
    _fields_ = [("classifierValueIds", POINTER(ctypes.c_size_t)),
                ("nClassifierValueIds", c_size_t),
                ("SoftwoodComponent", POINTER(LibCBM_MerchVolumeComponent)),
                ("HardwoodComponent", POINTER(LibCBM_MerchVolumeComponent))]

class LibCBM_CoordinateMatrix(ctypes.Structure):
    _fields_ = [("memsize", ctypes.c_size_t),
                ("count", ctypes.c_size_t),
                ("rows", POINTER(ctypes.c_size_t)),
                ("cols", POINTER(ctypes.c_size_t)),
                ("values", POINTER(ctypes.c_double))]

class LibCBMWrapper(object):
    def __init__(self, dllpath):
        self.handle = False
        self._dll = ctypes.CDLL(dllpath)

        self._dll.LibCBM_Free(
            ctypes.POINTER(LibCBM_Error),
            ctypes.c_void_p)

        self._dll.LibCBM_Initialize(
            ctypes.POINTER(LibCBM_Error),
            ctypes.c_size_t, # poolcount
            ctypes.POINTER(ctypes.c_char_p), # poolnames
            ctypes.c_char_p, # dbpath
            ctypes.c_size_t, # random seed
            ctypes.POINTER(LibCBM_Classifier), # classifiers
            ctypes.c_size_t, # number of classifiers
            ctypes.POINTER(LibCBM_ClassifierValue), # classifier values
            ctypes.c_size_t, # number of classifier values
            ctypes.POINTER(LibCBM_MerchVolumeCurve), # merch volume curves
            ctypes.c_size_t #number of merch vol curves
            )


