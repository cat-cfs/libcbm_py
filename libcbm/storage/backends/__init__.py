from enum import Enum


class BackendType(Enum):
    numpy = 1
    pandas = 2
    pyarrow = 3


def get_backend(backend_type: BackendType):

    if backend_type == BackendType.numpy:
        from libcbm.storage.backends import numpy_backend

        return numpy_backend
    elif backend_type == BackendType.pandas:
        from libcbm.storage.backends import pandas_backend

        return pandas_backend
    else:
        raise NotImplementedError()
