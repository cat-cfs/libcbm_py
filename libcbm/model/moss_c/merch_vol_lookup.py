from types import SimpleNamespace
import numpy as np
import numba.types


@numba.njit()
def _get_merch_volume(
    volume_lookup_dict, max_age_lookup_dict, age, merch_vol_id, output
):
    for i, age in np.ndenumerate(age):
        _id = merch_vol_id[i]
        lookup = volume_lookup_dict[_id]
        if age in lookup:
            output[i] = lookup[age]
        else:
            max_age = max_age_lookup_dict[_id]
            if age > max_age:
                output[i] = lookup[max_age]
            else:
                raise ValueError("age not defined")


class MerchVolumeLookup:
    def __init__(self, merch_volume):

        self._lookup = {
            int(i): SimpleNamespace(age_volume_pairs={}, max_age=0)
            for i in merch_volume.index
        }

        for _, row in merch_volume.iterrows():
            record = self._lookup[int(row.name)]
            volume = float(row.volume)
            age = int(row.age)
            if age < 0 or volume < 0:
                raise ValueError("negative age or volume found")
            if age >= record.max_age:
                record.max_age = age
            record.age_volume_pairs[age] = volume

        self._numba_lookup = numba.typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.types.DictType(
                numba.types.int64, numba.types.float64
            ),
        )

        self._numba_max_ages = numba.typed.Dict.empty(
            key_type=numba.types.int64, value_type=numba.types.int64
        )

        for i, record in self._lookup.items():
            if i not in self._numba_lookup:
                self._numba_max_ages[i] = record.max_age
                self._numba_lookup[i] = numba.typed.Dict.empty(
                    key_type=numba.types.int64, value_type=numba.types.float64
                )
            for age, volume in record.age_volume_pairs.items():
                self._numba_lookup[i][age] = volume

    def get_merch_vol(self, age, merch_vol_id):
        output = np.zeros(shape=age.shape, dtype=float)
        _get_merch_volume(
            self._numba_lookup, self._numba_max_ages, age, merch_vol_id, output
        )
        return output
