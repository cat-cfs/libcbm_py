import os
from typing import Union, Mapping, Any
import numpy as np
import pandas as pd
import ctypes
from enum import IntEnum
from libcbm import resources
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix
from libcbm.wrapper.libcbm_error import LibCBM_Error


class LibV2B_ConversionMode(IntEnum):

    CBM3 = 0
    """
    CBM3 mode produces the Softwood and Hardwood Merch Foliage, and Other
    output
    """

    Extended = 1
    """Extended produces more detailed outputs, including more of the
    components produced by the Boudewyn et al Volume to biomass routines.
    """

    ExtendedProportions = 2
    """Extended produces more detailed outputs, including more of the
    components produced by the Boudewyn et al Volume to biomass routines.
    This is the same as extended, but values are expressed as proportions
    of the CBM-CFS3 pool
    """


class LibV2B_MerchVolumeCurve(ctypes.Structure):

    _fields_ = [
        ("species_code", ctypes.c_int),
        ("size", ctypes.c_size_t),
        ("ages", ctypes.POINTER(ctypes.c_int)),
        ("merchvol", ctypes.POINTER(ctypes.c_double)),
    ]

    def __init__(self, species_code: int, ages: np.ndarray, merchvol: np.ndarray):
        self.species_code = species_code
        self.size = ages.shape[0]
        self.ages = ages.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.merchvol = merchvol.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class LibV2B_ConversionInfo(ctypes.Structure):
    _fields_ = [
        ("softwood_carbon_fraction_stemwood", ctypes.c_double),
        ("softwood_carbon_fraction_foliage", ctypes.c_double),
        ("softwood_carbon_fraction_other", ctypes.c_double),
        ("softwood_carbon_fraction_coarse_root", ctypes.c_double),
        ("softwood_carbon_fraction_fine_root", ctypes.c_double),
        ("hardwood_carbon_fraction_stemwood", ctypes.c_double),
        ("hardwood_carbon_fraction_foliage", ctypes.c_double),
        ("hardwood_carbon_fraction_other", ctypes.c_double),
        ("hardwood_carbon_fraction_coarse_root", ctypes.c_double),
        ("hardwood_carbon_fraction_fine_root", ctypes.c_double),
        ("softwood_leading_species", ctypes.c_double),
        ("hardwood_leading_species", ctypes.c_double),
    ]

    def __init__(self):
        self.softwood_carbon_fraction_stemwood = 0.0
        self.softwood_carbon_fraction_foliage = 0.0
        self.softwood_carbon_fraction_other = 0.0
        self.softwood_carbon_fraction_coarse_root = 0.0
        self.softwood_carbon_fraction_fine_root = 0.0
        self.hardwood_carbon_fraction_stemwood = 0.0
        self.hardwood_carbon_fraction_foliage = 0.0
        self.hardwood_carbon_fraction_other = 0.0
        self.hardwood_carbon_fraction_coarse_root = 0.0
        self.hardwood_carbon_fraction_fine_root = 0.0
        self.softwood_leading_species = 0
        self.hardwood_leading_species = 0

    def attrs(self) -> Mapping[str, Any]:
        return {f[0]: getattr(self, f[0]) for f in self._fields_}


class MerchVolumeCurve:
    def __init__(self, species_code: int, age: np.ndarray, merchvol: np.ndarray):
        """A component of a volume to biomass conversion

        Args:
            species_code (int): The species code of the component (see
                cbm_defaults database for definition)
            age (np.ndarray): The array of ages in years for each of the
                volumes
            merchvol (np.ndarray): The array of Merchantable volumes
        """
        assert age.dtype == np.int32, "Ages must be int32"
        assert merchvol.dtype == np.float64, "Volumes must be float64"
        self.species_code = species_code
        self.age = age
        self.merchvol = merchvol


class VolumeToBiomassWrapper:

    def __init__(self, dllpath: Union[str, None] = None):
        """Initialize the volume to biomass wrapper for running the CBM-CFS3
        volume to biomass conversion routine

        Args:
            dllpath (str | None, optional): Optional path to a dll, if not
                specified the libcbm bundled value is used. Defaults to None.
        """
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
            ctypes.POINTER(LibV2B_ConversionInfo),  # conversion info
            ctypes.c_int,  # conversion mode (enum)
            ctypes.POINTER(LibCBM_Error),  # err struct
        )

    def volume_to_biomass(
        self,
        spatial_unit_id: int,
        merch_vols: list[MerchVolumeCurve],
        use_smoother: bool = True,
        conversion_mode: LibV2B_ConversionMode = LibV2B_ConversionMode.CBM3,
        db_path: Union[str, None] = None,
    ) -> pd.DataFrame:
        """Convert a set of Age/Merchantable volume curves into CBM-CFS3 above
        ground biomass Carbon pools

        Args:
            spatial_unit_id (int): A spatial unit identifier as defined in the
                cbm_defaults database.
            merch_vols (list[MerchVolumeCurve]): A collection of one or more
            use_smoother (bool, optional): If set to true the CBM-CFS3
                "smoother" is enabled, and it is otherwise disabled. Defaults
                to True.
            conversion_mode (LibV2B_ConversionMode, optional): Can be set to
                produce more detailed output based on the volume to biomass
                routines. Defaults to LibV2B_ConversionMode.CBM3, which
                produces only the CBM-CFS3 above ground biomass Carbon pools.
            db_path (str | None, optional): Path to a cbm_defaults parameter
                database. In this context it is used to provide the relevant
                volume to biomass parameters and stump/top proportion
                parameters. Defaults to None.

        Raises:
            RuntimeError: An error occurred in the dll call

        Returns:
            pd.DataFrame: the result of the conversion:

                * When LibV2B_ConversionMode.CBM3 this has the 6 CBM-CFS3
                  above ground biomass C columns, and 1 row from age 1 to
                  max(age)
                * When the extended modes are used, there are 23 columns, and
                  the same number of rows

        """

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
        if len(merch_vols) == 0:
            return pd.DataFrame(columns=out_cols)
        merch_vols_array = (LibV2B_MerchVolumeCurve * len(merch_vols))()

        for i_merch_vol, merch_vol in enumerate(merch_vols):
            merch_vols_array[i_merch_vol] = LibV2B_MerchVolumeCurve(
                merch_vol.species_code, merch_vol.age, merch_vol.merchvol
            )
        merch_vols_p = ctypes.cast(
            merch_vols_array, ctypes.POINTER(LibV2B_MerchVolumeCurve)
        )

        max_age: int = max([m.age.max() for m in merch_vols])
        out_data = np.zeros(shape=(max_age, len(out_cols)))
        err = LibCBM_Error()
        conversion_info = LibV2B_ConversionInfo()
        self._dll.VolumeToBiomass(
            ctypes.c_char_p(db_path.encode("UTF-8")),
            spatial_unit_id,
            merch_vols_p,
            len(merch_vols),
            LibCBM_Matrix(out_data),
            int(use_smoother),
            conversion_info,
            int(conversion_mode),
            ctypes.byref(err),
        )
        if err.Error != 0:
            raise RuntimeError(err.getErrorMessage())
        result = pd.DataFrame(columns=out_cols, data=out_data)
        result.attrs["conversion_info"] = conversion_info.attrs()
        return result
