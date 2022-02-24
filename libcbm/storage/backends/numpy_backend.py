import ctypes
import numpy as np


def get_numpy_pointer(
    data: np.ndarray, dtype=ctypes.c_double
) -> ctypes.pointer:
    """Helper method for wrapper parameters that can be specified either as
    null pointers or pointers to numpy memory.  Return a pointer to float64
    or int32 memory for use with ctypes wrapped functions, or None if None
    is specified.

    Args:
        data (numpy.ndarray, None): array to convert to pointer, if None is
            specified None is returned.
        type (object, optional): type supported by ctypes.POINTER. Defaults
            to ctypes.c_double.  Since libcbm only currently uses int32, or
            float 64, the only valid values are those that equal
            ctypes.c_double, or ctypes.c_int32

    Returns:
        None or ctypes.pointer: if the specified argument is None, None is
            returned, otherwise the argument is converted to a pointer to
            the underlying ndarray data.
    """
    if data is None:
        return None
    else:

        if not data.flags["C_CONTIGUOUS"]:
            raise ValueError("specified array is not C_CONTIGUOUS")
        if dtype == ctypes.c_double:
            if data.dtype != np.dtype("float64"):
                raise ValueError(
                    f"specified array is of type {data.dtype} "
                    f"and cannot be converted to {dtype}."
                )
        elif dtype == ctypes.c_int32:
            if data.dtype != np.dtype("int32"):
                raise ValueError(
                    f"specified array is of type {data.dtype} "
                    f"and cannot be converted to {dtype}."
                )
        else:
            raise ValueError(f"unsupported type {dtype}")
        p_result = data.ctypes.data_as(ctypes.POINTER(dtype))
        return p_result
