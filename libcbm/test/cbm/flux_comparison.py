import pandas as pd
import collections
from libcbm.test.cbm import result_comparison


def get_libcbm_flux_disturbance_cols():
    """Returns the ordered set of names for LibCBM columns involved in
    disturbance fluxes.

    Returns:
        list: a list of strings which are the ordered LibCBM flux indicator
        columns involved in disturbance fluxes.
    """
    return [
        'DisturbanceCO2Production',
        'DisturbanceCH4Production',
        'DisturbanceCOProduction',
        'DisturbanceBioCO2Emission',
        'DisturbanceBioCH4Emission',
        'DisturbanceBioCOEmission',
        'DisturbanceSoftProduction',
        'DisturbanceHardProduction',
        'DisturbanceDOMProduction',
        'DisturbanceMerchToAir',
        'DisturbanceFolToAir',
        'DisturbanceOthToAir',
        'DisturbanceCoarseToAir',
        'DisturbanceFineToAir',
        'DisturbanceDOMCO2Emission',
        'DisturbanceDOMCH4Emssion',
        'DisturbanceDOMCOEmission',
        'DisturbanceMerchLitterInput',
        'DisturbanceFolLitterInput',
        'DisturbanceOthLitterInput',
        'DisturbanceCoarseLitterInput',
        'DisturbanceFineLitterInput',
        'DisturbanceVFastAGToAir',
        'DisturbanceVFastBGToAir',
        'DisturbanceFastAGToAir',
        'DisturbanceFastBGToAir',
        'DisturbanceMediumToAir',
        'DisturbanceSlowAGToAir',
        'DisturbanceSlowBGToAir',
        'DisturbanceSWStemSnagToAir',
        'DisturbanceSWBranchSnagToAir',
        'DisturbanceHWStemSnagToAir',
        'DisturbanceHWBranchSnagToAir'
    ]


def get_libcbm_flux_annual_process_cols():
    """Returns the ordered set of names for LibCBM columns involved in
    annual process fluxes.

    Returns:
        list: a list of strings which are the ordered LibCBM flux indicator
        columns involved in annual process fluxes.
    """
    return [
        'DecayDOMCO2Emission',
        'DeltaBiomass_AG',
        'DeltaBiomass_BG',
        'TurnoverMerchLitterInput',
        'TurnoverFolLitterInput',
        'TurnoverOthLitterInput',
        'TurnoverCoarseLitterInput',
        'TurnoverFineLitterInput',
        'DecayVFastAGToAir',
        'DecayVFastBGToAir',
        'DecayFastAGToAir',
        'DecayFastBGToAir',
        'DecayMediumToAir',
        'DecaySlowAGToAir',
        'DecaySlowBGToAir',
        'DecaySWStemSnagToAir',
        'DecaySWBranchSnagToAir',
        'DecayHWStemSnagToAir',
        'DecayHWBranchSnagToAir'
    ]


def get_cbm3_flux_disturbance_cols():
    """Returns the ordered set of names for CBM3 columns involved in
    disturbance fluxes.

    Returns:
        list: a list of strings which are the ordered CBM3 columns involved in
        disturbance fluxes.
    """
    return[
        'CO2Production',
        'CH4Production',
        'COProduction',
        'BioCO2Emission',
        'BioCH4Emission',
        'BioCOEmission',
        'SoftProduction',
        'HardProduction',
        'DOMProduction',
        'MerchToAir',
        'FolToAir',
        'OthToAir',
        'CoarseToAir',
        'FineToAir',
        'DOMCO2Emission',
        'DOMCH4Emssion',
        'DOMCOEmission',
        'MerchLitterInput',
        'FolLitterInput',
        'OthLitterInput',
        'CoarseLitterInput',
        'FineLitterInput',
        'VFastAGToAir',
        'VFastBGToAir',
        'FastAGToAir',
        'FastBGToAir',
        'MediumToAir',
        'SlowAGToAir',
        'SlowBGToAir',
        'SWStemSnagToAir',
        'SWBranchSnagToAir',
        'HWStemSnagToAir',
        'HWBranchSnagToAir',
    ]


def get_cbm3_flux_annual_process_cols():
    """Returns the ordered set of names for CBM3 columns involved in annual
    process fluxes.

    Returns:
        list: a list of strings which are the ordered CBM3 columns involved
        in annual process fluxes.
    """
    return [
        'DOMCO2Emission',
        'DeltaBiomass_AG',
        'DeltaBiomass_BG',
        'MerchLitterInput',
        'FolLitterInput',
        'OthLitterInput',
        'CoarseLitterInput',
        'FineLitterInput',
        'VFastAGToAir',
        'VFastBGToAir',
        'FastAGToAir',
        'FastBGToAir',
        'MediumToAir',
        'SlowAGToAir',
        'SlowBGToAir',
        'SWStemSnagToAir',
        'SWBranchSnagToAir',
        'HWStemSnagToAir',
        'HWBranchSnagToAir'
    ]


def get_cbm3_disturbance_flux(cbm3_flux):
    """Returns disturbance flux from a query result of CBM3 flux indicators.

    Also performs the following table changes to make it easy to join and
    compare with to the libcbm result:

        - rename "TimeStep" to "timestep"
        - convert the "identifier" column to numeric from string
        - rename the columns according to the :py:func:`zip` of
            :py:func:`get_cbm3_flux_disturbance_cols` to
            :py:func:`get_libcbm_flux_disturbance_cols`
        - negate the values for the following columns due to a quirk in
            CBM-CFS3 flux indicator results:
            - 'MerchToAir'
            - 'FolToAir'
            - 'OthToAir'
            - 'CoarseToAir'
            - 'FineToAir'

    Args:
        cbm3_flux (pandas.DataFrame): The CBM-CFS3 disturbance flux

    Returns:
        pandas.DataFrame: a filtered and altered copy of the input containing
            only disturbance fluxes.
    """

    # in CBM3 a default disturbance type of 0 indicates annual process
    # in tblFluxIndicators rows by convention.
    flux = cbm3_flux.loc[cbm3_flux["DefaultDistTypeID"] != 0].copy()
    flux = flux.rename(columns={'TimeStep': 'timestep'})
    flux["identifier"] = pd.to_numeric(flux["identifier"])

    # account for a CBM-CFS3 quirk where the following fluxes are negated
    # in the tblFluxIndicators results.
    biomass_to_air_cols = [
        'MerchToAir', 'FolToAir', 'OthToAir', 'CoarseToAir', 'FineToAir']
    for b in biomass_to_air_cols:
        flux[b] = flux[b] * -1.0

    disturbance_flux_mapping = collections.OrderedDict(
        zip(
            get_cbm3_flux_disturbance_cols(),
            get_libcbm_flux_disturbance_cols()))

    flux = flux.rename(columns=disturbance_flux_mapping)
    return flux


def get_cbm3_annual_process_flux(cbm3_flux):
    """Returns annual process flux from a query result of CBM3 flux
    indicators.

    Also performs the following table changes to make it easy to join and
    compare with to the libcbm result:

        - rename "TimeStep" to "timestep"
        - convert the "identifier" column to numeric from string
        - rename the columns according to the :py:func:`zip` of
            :py:func:`get_cbm3_flux_disturbance_cols` to
            :py:func:`get_libcbm_flux_annual_process_cols`

    Args:
        cbm3_flux (pandas.DataFrame): The CBM-CFS3 flux indicator result

    Returns:
        pandas.DataFrame: a filtered and altered copy of the input containing
            only annual process fluxes.
    """

    # in CBM3 a default disturbance type of 0 indicates annual process
    # in tblFluxIndicators rows by convention.
    flux = cbm3_flux.loc[cbm3_flux["DefaultDistTypeID"] == 0].copy()
    flux = flux.rename(columns={'TimeStep': 'timestep'})
    flux["identifier"] = pd.to_numeric(flux["identifier"])

    disturbance_flux_mapping = collections.OrderedDict(
        zip(
            get_cbm3_flux_annual_process_cols(),
            get_libcbm_flux_annual_process_cols()))

    flux = flux.rename(columns=disturbance_flux_mapping)
    return flux


def get_merged_annual_process_flux(cbm3_flux, libcbm_flux):
    merged_pools = result_comparison.merge_result(
        get_cbm3_annual_process_flux(cbm3_flux),
        libcbm_flux,
        get_libcbm_flux_annual_process_cols())
    return merged_pools
