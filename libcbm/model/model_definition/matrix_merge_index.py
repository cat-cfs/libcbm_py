from typing import Callable
from typing import Union
import numpy as np
import numba
import numba.typed


@numba.njit(inline="always")
def k1(i, m):
    return (m[0][i],)


@numba.njit(inline="always")
def k2(i, m):
    return (m[0][i], m[1][i])


@numba.njit(inline="always")
def k3(i, m):
    return (m[0][i], m[1][i], m[2][i])


@numba.njit(inline="always")
def k4(i, m):
    return (m[0][i], m[1][i], m[2][i], m[3][i])


@numba.njit(inline="always")
def k5(i, m):
    return (m[0][i], m[1][i], m[2][i], m[3][i], m[4][i])


@numba.njit(inline="always")
def k6(i, m):
    return (m[0][i], m[1][i], m[2][i], m[3][i], m[4][i], m[5][i])


@numba.njit(inline="always")
def k7(i, m):
    return (m[0][i], m[1][i], m[2][i], m[3][i], m[4][i], m[5][i], m[6][i])


@numba.njit(inline="always")
def k8(i, m):
    return (
        m[0][i],
        m[1][i],
        m[2][i],
        m[3][i],
        m[4][i],
        m[5][i],
        m[6][i],
        m[7][i],
    )


KEY_FUNC_MAP = {1: k1, 2: k2, 3: k3, 4: k4, 5: k5, 6: k6, 7: k7, 8: k8}


def get_key_func(size: int) -> Callable:
    return KEY_FUNC_MAP[size]


@numba.njit()
def merge(
    k_func,
    out,
    merge_dict: numba.typed.Dict,
    len_merge_arrays: int,
    fill: int,
    error_on_missing: bool,
    *merge_arrays,
):
    for i in range(len_merge_arrays):
        k = k_func(i, merge_arrays)
        if k in merge_dict:
            out[i] = merge_dict[k]
        else:
            if error_on_missing:
                return i
            out[i] = fill
    return -1


class MatrixMergeIndex:
    def __init__(
        self, key_data: dict[str, np.ndarray], matrix_idx: np.ndarray
    ):
        self._merge_keys = list(key_data.keys())
        self._key_data = key_data
        self._len_key_data = len((next(iter(key_data.values()))))

        key_index_type = numba.types.UniTuple(
            numba.types.float64, len(self._merge_keys)
        )
        self._merge_dict = numba.typed.Dict.empty(
            key_type=key_index_type, value_type=numba.types.types.uint64
        )
        for i in range(self._len_key_data):
            tuple_values = []
            for k in self._merge_keys:
                tuple_values.append(float(key_data[k][i]))
            self._merge_dict[tuple(tuple_values)] = matrix_idx[i]

    @property
    def merge_keys(self) -> list[str]:
        return self._merge_keys.copy()

    def merge(
        self,
        merge_data: dict[str, np.ndarray],
        fill_value: Union[int, None] = None,
    ) -> np.ndarray:
        len_merge_arrays = len((next(iter(merge_data.values()))))
        out = np.empty(len_merge_arrays, dtype="int64")
        err_idx = merge(
            get_key_func(len(self._merge_keys)),
            out,
            self._merge_dict,
            len_merge_arrays,
            fill_value if fill_value is not None else -1,
            fill_value is None,
            *list(merge_data.values()),
        )
        if err_idx > 0:
            values_not_found = [v[err_idx] for v in merge_data.values()]
            raise ValueError(f"did not find values for {values_not_found}")
        return out
