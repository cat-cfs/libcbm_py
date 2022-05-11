from libcbm.storage.backends import BackendType


def get_backend(backend_type: BackendType):
    if backend_type == BackendType.pandas:
        from libcbm.storage.backends import pandas_backend

        return pandas_backend
