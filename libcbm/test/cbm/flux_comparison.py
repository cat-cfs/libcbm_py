import pandas as pd


def get_cbm3_disturbance_flux(cbm3_flux):
    """Returns disturbance flux from a query result of CBM3 flux indicators.

    Also performs the following table changes to make it easy to join to the
    libcbm result.

        - rename "TimeStep" to "timestep"
        - convert the "identifier" column to numeric from string

    Args:
        cbm3_flux (pandas.DataFrame): The CBM-CFS3 disturbance flux

    Returns:
        pandas.DataFrame: a filtered and altered copy of the input
    """
    flux = cbm3_flux.loc[cbm3_flux["DefaultDistTypeID"] != 0].copy()
    flux = flux.rename(columns={'TimeStep': 'timestep'})
    flux["identifier"] = pd.to_numeric(flux["identifier"])
    return flux


def join_disturbance_flux(cbm3_disturbance_flux):

    cbm3_flux = cbm3_flux.rename(columns={'TimeStep': 'timestep'})
    cbm3_flux["identifier"] = pd.to_numeric(cbm3_flux["identifier"])