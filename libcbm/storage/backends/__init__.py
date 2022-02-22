from enum import Enum


class BackendType(Enum):
    numpy = 1
    pandas = 2
    pyarrow = 3
