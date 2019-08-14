import pandas as pd
import collections


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
    """[summary]

    Returns:
        [type]: [description]
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
    compare with to the libcbm result.

        - rename "TimeStep" to "timestep"
        - convert the "identifier" column to numeric from string
        - rename the columns according to the :py:func:`zip` of
        :py:func`get_cbm3_flux_disturbance_cols` to

    Args:
        cbm3_flux (pandas.DataFrame): The CBM-CFS3 disturbance flux

    Returns:
        pandas.DataFrame: a filtered and altered copy of the input
    """
    flux = cbm3_flux.loc[cbm3_flux["DefaultDistTypeID"] != 0].copy()
    flux = flux.rename(columns={'TimeStep': 'timestep'})
    flux["identifier"] = pd.to_numeric(flux["identifier"])

    disturbance_flux_mapping = collections.OrderedDict(
        zip(
            get_cbm3_flux_disturbance_cols(),
            get_libcbm_flux_disturbance_cols()))

    flux = flux.rename(columns=disturbance_flux_mapping)
    return flux


def join_disturbance_flux(cbm3_disturbance_flux):

    cbm3_flux = cbm3_flux.rename(columns={'TimeStep': 'timestep'})
    cbm3_flux["identifier"] = pd.to_numeric(cbm3_flux["identifier"])