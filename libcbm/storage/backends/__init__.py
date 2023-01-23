from enum import Enum


class BackendType(Enum):
    """Enumeration of the supported dataFrame, series backend types in libcbm"""

    numpy = 1
    """the numpy backend type
    """

    pandas = 2
    """the pandas backend type
    """
    # dask = 3
    # pyarrow = 3


def get_backend(backend_type: BackendType):
    """get the implementation of a backend type

    Args:
        backend_type (BackendType): one of the supported types

    Raises:
        NotImplementedError: the specified value is not supported or
            implemented

    Returns:
        module: the backend
    """
    if backend_type == BackendType.numpy:
        from libcbm.storage.backends import numpy_backend

        return numpy_backend
    elif backend_type == BackendType.pandas:
        from libcbm.storage.backends import pandas_backend

        return pandas_backend
    else:
        raise NotImplementedError()
