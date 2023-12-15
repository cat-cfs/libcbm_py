from typing import Callable
from typing import Union
import numpy as np
import numba
import numba.typed
from libcbm.model.model_definition.model_variables import ModelVariables


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
    """
    Creates and stores an index for indexed matrices. This is used to
    efficiently merge the matrix information to each simulation area during
    runtime.
    """

    def __init__(
        self, nrows: int, key_data: Union[dict[str, np.ndarray], None]
    ):
        """Intialize a MatrixMergeIndex

        Args:
            key_data (dict[str, np.ndarray]): the key data for each matrix

        Raises:
            ValueError: only integer type keys are supported
        """
        self._len_key_data = nrows
        if key_data:
            self._merge_keys = list(key_data.keys())
            self._key_data = key_data
            # assumption here is that all members of key_data are of equal
            # length
            for i_v, v in enumerate(key_data.values()):
                if v.ndim > 1:
                    raise ValueError("expected single dimensional key values")

                if self._len_key_data != v.shape[0]:
                    raise ValueError("lengths of key data array non-uniform")

            key_index_type = numba.types.UniTuple(
                numba.types.int64, len(self._merge_keys)
            )
            self._merge_dict = numba.typed.Dict.empty(
                key_type=key_index_type, value_type=numba.types.types.uint64
            )
            for i in range(self._len_key_data):
                tuple_values = []
                for k in self._merge_keys:
                    key_val = int(key_data[k][i])
                    if key_val != key_data[k][i]:
                        raise ValueError(
                            "only integer keys supported. Found: "
                            f"{key_data[k][i]} in {k} series"
                        )
                    tuple_values.append(key_val)
                self._merge_dict[tuple(tuple_values)] = np.uint64(i)
        else:
            self._merge_keys = []
            self._key_data = {}
            self._merge_dict = {}

    @property
    def has_keys(self) -> bool:
        return len(self._merge_keys) > 0

    @property
    def merge_keys(self) -> list[str]:
        """gets a copy of the merge keys for this instance

        Returns:
            list[str]: the merge keys
        """
        return self._merge_keys.copy()

    def compute_matrix_index(
        self,
        model_variables: ModelVariables,
        default_matrix_index: Union[int, None],
    ) -> np.ndarray:
        n_rows = model_variables["pools"].n_rows

        if not self.has_keys:
            if self._len_key_data == 1:
                # project the single operation to the entire landscape
                return np.full(n_rows, 0, dtype="uintp")
            elif self._len_key_data == n_rows:
                # there is one operation for each simulation area
                return np.arange(0, n_rows, dtype="uintp")
            else:
                raise ValueError(
                    "index length must match model_variables length, or be "
                    "of length 1."
                )
        else:
            merge_data = {}
            for idx_name in self.merge_keys:
                if idx_name == "row_idx":
                    merge_data["row_idx"] = np.arange(0, n_rows, dtype="int64")
                else:
                    s = idx_name.split(".")
                    merge_data[idx_name] = (
                        model_variables[s[0]][s[1]].to_numpy().astype("int64")
                    )

            return self.merge(merge_data, default_matrix_index)

    def merge(
        self,
        merge_data: dict[str, np.ndarray],
        fill_value: Union[int, None] = None,
    ) -> np.ndarray:
        """Merge the index values stored in this instance to the specified
        merge data.

        Args:
            merge_data (dict[str, np.ndarray]): Values to merge with this
                instance's stored key values.
            fill_value (Union[int, None], optional): An optional fill value.
                This is used to fill values where no matching key is found
                in this instance's index. If this value is unspecifed, missing
                any missing keys will result in a value error. Defaults to
                None.

        Raises:
            ValueError: Raised if fill value is not set and 1 or more value in
                the specified merge data is not found in this instance's
                stored index.

        Returns:
            np.ndarray: the index of each matched key for each element in the
                specified merge data
        """
        if fill_value is not None:
            if fill_value < 0 or fill_value >= self._len_key_data:
                raise ValueError(
                    "fill must be within the range of defined "
                    f"key indexes (0,{self._len_key_data-1}). "
                    f"got: {fill_value}"
                )
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
        if err_idx >= 0:
            values_not_found = {k: v[err_idx] for k, v in merge_data.items()}
            raise ValueError(f"did not find values for {values_not_found}")
        return out
