import numpy as np
from libcbm.wrapper.volume_to_biomass import volume_to_biomass_wrapper
from libcbm.wrapper.volume_to_biomass.volume_to_biomass_wrapper import (
    MerchVolumeCurve,
    LibV2B_ConversionMode,
)


def test_volume_to_biomass_wrapper_functional():
    wrapper = volume_to_biomass_wrapper.VolumeToBiomassWrapper()
    result = wrapper.volume_to_biomass(
        spatial_unit_id=42,
        merch_vols=[
            MerchVolumeCurve(
                species_code=1,
                age=np.array([0, 10, 20], dtype="int32"),
                merchvol=np.array([0, 0.1, 0.5]),
            )
        ],
        conversion_mode=LibV2B_ConversionMode.CBM3,
    )
    assert len(result.columns) == 7
    assert len(result.index) == 20
    assert result.attrs["softwood_leading_species"] == 1
    assert result.attrs["hardwood_leading_species"] == -1


def test_volume_to_biomass_wrapper_smooth_disable():
    wrapper = volume_to_biomass_wrapper.VolumeToBiomassWrapper()
    result = wrapper.volume_to_biomass(
        spatial_unit_id=42,
        merch_vols=[
            MerchVolumeCurve(
                species_code=1,
                age=np.array([0, 10, 20], dtype="int32"),
                merchvol=np.array([0, 0.1, 0.5]),
            )
        ],
        conversion_mode=LibV2B_ConversionMode.CBM3,
        use_smoother=False,
    )
    assert len(result.columns) == 7
    assert len(result.index) == 20


def test_volume_to_biomass_wrapper_smooth_extended_mode():
    wrapper = volume_to_biomass_wrapper.VolumeToBiomassWrapper()
    result = wrapper.volume_to_biomass(
        spatial_unit_id=42,
        merch_vols=[
            MerchVolumeCurve(
                species_code=1,
                age=np.array([0, 10, 20], dtype="int32"),
                merchvol=np.array([0, 0.1, 0.5]),
            )
        ],
        conversion_mode=LibV2B_ConversionMode.Extended,
        use_smoother=False,
    )
    assert len(result.columns) == 23
    assert len(result.index) == 20


def test_volume_to_biomass_wrapper_smooth_extended_proportion_mode():
    wrapper = volume_to_biomass_wrapper.VolumeToBiomassWrapper()
    result = wrapper.volume_to_biomass(
        spatial_unit_id=42,
        merch_vols=[
            MerchVolumeCurve(
                species_code=1,
                age=np.array([0, 10, 20], dtype="int32"),
                merchvol=np.array([0, 0.1, 0.5]),
            )
        ],
        conversion_mode=LibV2B_ConversionMode.Extended,
        use_smoother=False,
    )
    assert len(result.columns) == 23
    assert len(result.index) == 20
