import os
import numpy as np
import pandas as pd
import ctypes
from enum import IntEnum
from libcbm import resources
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix
from libcbm.wrapper.libcbm_error import LibCBM_Error


class LibV2B_ConversionMode(IntEnum):
    CBM3 = 0
    Extended = 1
    ExtendedProportions = 2


class LibV2B_MerchVolumeCurve(ctypes.Structure):

    _fields_ = [
        ("species_code", ctypes.c_int),
        ("size", ctypes.c_size_t),
        ("ages", ctypes.POINTER(ctypes.c_int)),
        ("merchvol", ctypes.POINTER(ctypes.c_double)),
    ]

    def __init__(
        self,
        species_code: int,
        ages: np.ndarray,
        merchvol: np.ndarray
    ):
        self.species_code = species_code
        self.size = ages.shape[0]
        self.ages = ages.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.merchvol = merchvol.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )


class MerchVolumeCurve:
    def __init__(
        self,
        species_code: int,
        age: np.ndarray,
        merchvol: np.ndarray
    ):
        self.species_code = species_code
        self.age = age
        self.merchvol = merchvol


class VolumeToBiomassWrapper:

    def __init__(self, dllpath: str | None = None):
        if dllpath is None:
            self._dllpath = resources.get_libcbm_bin_path()
        else:
            self._dllpath = dllpath

        cwd = os.getcwd()
        os.chdir(os.path.dirname(self._dllpath))
        self._dll = ctypes.CDLL(self._dllpath)
        os.chdir(cwd)

        self._dll.VolumeToBiomass.argtypes = (
            ctypes.c_char_p,  # db path
            ctypes.c_int,  # spatial unit id
            ctypes.POINTER(LibV2B_MerchVolumeCurve),  # merch volumes
            ctypes.c_size_t,  # n merch volumes
            LibCBM_Matrix,  # biomass carbon (output)
            ctypes.c_int,  # useSmoother (bool)
            ctypes.c_int,  # conversion mode (enum)
            ctypes.POINTER(LibCBM_Error),  # err struct
        )

    def volume_to_biomass(
        self,
        spatial_unit_id: int,
        merch_vols: list[MerchVolumeCurve],
        use_smoother: bool = True,
        conversion_mode: LibV2B_ConversionMode = LibV2B_ConversionMode.CBM3,
        db_path: str | None = None,
    ) -> pd.DataFrame:

        if db_path is None:
            db_path = resources.get_cbm_defaults_path()
        out_cols = [
            "Age",
            "SWMerch",
            "SWFoliage",
            "SWOther",
            "HWMerch",
            "HWFoliage",
            "HWOther",
        ]

        if conversion_mode in [1, 2]:
            out_cols.extend(
                [
                    "SWMerchBark",
                    "SWMerchStemwood",
                    "SWOtherBark",
                    "SWOtherBranch",
                    "SWOtherSapling",
                    "SWOtherNonmerch",
                    "SWOtherStump",
                    "SWOtherTop",
                    "HWMerchBark",
                    "HWMerchStemwood",
                    "HWOtherBark",
                    "HWOtherBranch",
                    "HWOtherSapling",
                    "HWOtherNonmerch",
                    "HWOtherStump",
                    "HWOtherTop",
                ]
            )
        merch_vols_array = (LibV2B_MerchVolumeCurve * len(merch_vols))()
        for i_merch_vol, merch_vol in enumerate(merch_vols):
            merch_vols_array[i_merch_vol] = LibV2B_MerchVolumeCurve(
                merch_vol.species_code,
                merch_vol.age,
                merch_vol.merchvol
            )
        merch_vols_p = ctypes.cast(
            merch_vols_array, ctypes.POINTER(LibV2B_MerchVolumeCurve)
        )

        max_age: int = max([m.age.max() for m in merch_vols])
        out_data = np.zeros(shape=(max_age, len(out_cols)))
        err = LibCBM_Error()

        self._dll.VolumeToBiomass(
            ctypes.c_char_p(db_path.encode("UTF-8")),
            spatial_unit_id,
            merch_vols_p,
            len(merch_vols),
            LibCBM_Matrix(out_data),
            int(use_smoother),
            int(conversion_mode),
            ctypes.byref(err)
        )
        if err.Error != 0:
            raise RuntimeError(err.getErrorMessage())
        result = pd.DataFrame(
            columns=out_cols,
            data=out_data
        )
        return result
