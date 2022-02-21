# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import ctypes


class LibCBM_Error(ctypes.Structure):
    """Wrapper for low level C/C++ LibCBM structure of the same name.
    Stores string error message when they occur in library functions.
    """

    _fields_ = [
        ("Error", ctypes.c_int),
        ("Message", ctypes.ARRAY(ctypes.c_byte, 1000)),
    ]

    def __init__(self):
        setattr(self, "Error", 0)
        setattr(self, "Message", ctypes.ARRAY(ctypes.c_byte, 1000)())

    def getError(self) -> int:
        """Gets the error code from an error returned by a library function.
        If no error occurred this is zero.
        """
        code = getattr(self, "Error")
        return code

    def getErrorMessage(self) -> str:
        """Gets the error message from an error returned by a library
        function.  If no error occurred this is an empty string.
        """
        msg = ctypes.cast(getattr(self, "Message"), ctypes.c_char_p).value
        return msg
