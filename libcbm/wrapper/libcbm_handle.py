import ctypes
from libcbm.wrapper.libcbm_error import LibCBM_Error
from libcbm.wrapper.libcbm_ctypes import LibCBM_ctypes


class LibCBMHandle(LibCBM_ctypes):

    def __init__(self, config):
        self.err = LibCBM_Error()
        self.pointer = self._dll.LibCBM_Initialize(
            config, ctypes.byref(self.err))
        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """frees the allocated libcbm handle"""
        if self.pointer:
            err = LibCBM_Error()
            self._dll.LibCBM_Free(ctypes.byref(err), self.pointer)
            if err.Error != 0:
                raise RuntimeError(err.getErrorMessage())

    def call(self, func_name, *args):
        func = getattr(self._dll, func_name)
        args = (ctypes.byref(self.err), self.handle) + args
        result = func(*args)
        if self.err.Error != 0:
            raise RuntimeError(self.err.getErrorMessage())
        return result
