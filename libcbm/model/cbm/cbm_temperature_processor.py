from __future__ import annotations
import pandas as pd
from libcbm.model.cbm.cbm_variables import CBMVariables
from libcbm.model.cbm import cbm_variables
from libcbm.storage.dataframe import DataFrame
from libcbm.storage.series import Series


class SpatialUnitMeanAnnualTemperatureProcessor:
    def __init__(self, mean_annual_temp_lookup: pd.DataFrame):
        """initialize a class to manage and assign mean annual temperatures
        during CBM simulation

        Duplicate values of timestep, spatial unit will result in a ValueError

        Args:
            mean_annual_temp_lookup (pd.DataFrame): a dataframe containing
                columns:
                    - timestep: the timestep of the temperature
                    - spatial_unit: the spatial unit of the temperature
                    - mean_annual_temp: the mean annual temperature
        """
        self._mean_annual_temp_lookup = mean_annual_temp_lookup

        # lookup of mean annual temperature by spatial unit (inner key) by
        # timestep (outer key)
        self._timestep_lookups: dict[int, dict[int, float]] = {}

        timesteps = mean_annual_temp_lookup["timestep"].to_list()
        spatial_units = mean_annual_temp_lookup["spatial_unit"].to_list()
        temperatures = mean_annual_temp_lookup["mean_annual_temp"].to_list()
        for idx in range(0, len(timesteps)):
            timestep = int(timesteps[idx])
            spatial_unit = int(spatial_units[idx])
            temperature = int(temperatures[idx])
            if timestep in self._timestep_lookups:
                if spatial_unit in self._timestep_lookups[timestep]:
                    raise ValueError(
                        "duplicate (timestep, spatial_unit) combination "
                        f"detected {(timestep, spatial_unit)}"
                    )
                else:
                    self._timestep_lookups[timestep][
                        spatial_unit
                    ] = temperature
            else:
                self._timestep_lookups[timestep] = {spatial_unit: temperature}

    def get_spinup_parameters(
        self,
        inventory: DataFrame,
        return_interval: Series = None,
        min_rotations: Series = None,
        max_rotations: Series = None,
    ) -> DataFrame:
        """Gets a spinup parameters dataframe with mean annual
        temperature set, and optionally other spinup parameters
        as well.

        The mean annual temperature values

        Args:
            inventory (DataFrame): inventory dataframe
            return_interval (Series, optional): The number of
                years between historical disturbances in the spinup function.
                Defaults to None.
            min_rotations (Series, optional): The minimum number
                of historical rotations to perform. Defaults to None.
            max_rotations (Series, optional): The maximum number
                of historical rotations to perform. Defaults to None.

        Returns:
            DataFrame: initialized spinup parameter dataframe
        """
        spinup_mean_annual_temp = self._timestep_lookups[0]
        return cbm_variables.initialize_spinup_parameters(
            inventory.n_rows,
            inventory.backend_type,
            return_interval=return_interval,
            min_rotations=min_rotations,
            max_rotations=max_rotations,
            mean_annual_temp=inventory["spatial_unit"].map(
                spinup_mean_annual_temp
            ),
        )

    def set_timestep_mean_annual_temperature(
        self, timestep: int, cbm_vars: CBMVariables
    ) -> CBMVariables:
        # assert that the timestep is defined in the temperature data
        if timestep not in self._timestep_lookups:
            raise ValueError(
                f"no mean annual temperature data for t={timestep}."
            )

        timestep_data: dict[int, float] = self._timestep_lookups[timestep]
        mapped_data = cbm_vars.inventory["spatial_unit"].map(timestep_data)
        mapped_data.name = "mean_annual_temp"
        if "mean_annual_temp" not in cbm_vars.parameters.columns:
            cbm_vars.parameters.add_column(mapped_data)
        cbm_vars.parameters["mean_annual_temp"].assign(
            mapped_data
        )
        return cbm_vars
