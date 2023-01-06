from __future__ import annotations
from copy import deepcopy
from typing import Callable
from libcbm.input.sit.sit_reader import SITData
from libcbm.input.sit.sit_cbm_defaults import SITCBMDefaults
from libcbm.input.sit.sit_mapping import SITMapping
from libcbm.input.sit.sit_cbm_config import SITIdentifierMapping


class SIT:
    def __init__(
        self,
        config: dict,
        sit_data: SITData,
        sit_defaults: SITCBMDefaults,
        sit_mapping: SITMapping,
        sit_identifier_mapping: SITIdentifierMapping,
    ):
        self._config = config
        # make copy due so that this class can safely modify the
        # underlying data without side effects
        self._sit_data = deepcopy(sit_data)
        self._mapping = sit_mapping
        self._defaults = sit_defaults
        self._sit_identifier_mapping = sit_identifier_mapping

    @property
    def config(self) -> dict:
        return self._config

    @property
    def defaults(self) -> SITCBMDefaults:
        return self._defaults

    @property
    def sit_data(self) -> SITData:
        return self._sit_data

    @property
    def sit_mapping(self) -> SITMapping:
        return self._mapping

    @property
    def default_disturbance_id_map(self) -> dict[int, int]:
        """
        Get map of sit disturbance type id to the default disturbance
        type id defined in cbm defaults database
        """
        return self._sit_identifier_mapping.default_disturbance_id_map.copy()

    @property
    def default_disturbance_name_map(self) -> dict[int, str]:
        """
        Get map of sid disturbance type id to the default disturbance
        type name defined in cbm defaults database
        """
        return self._sit_identifier_mapping.default_disturbance_name_map.copy()

    @property
    def disturbance_id_map(self) -> dict[int, str]:
        """
        Get a map of the sit_disturbances 1 based sequential row id to the
        sit name/id (sit_disturbances col 0)
        """
        return self._sit_identifier_mapping.disturbance_id_map.copy()

    @property
    def disturbance_name_map(self) -> dict[int, str]:
        """
        Get a map of the sit_disturbances 1 based sequential row id to the sit
        description (sit_disturbances col 1)
        """
        return self._sit_identifier_mapping.disturbance_name_map.copy()

    @property
    def classifier_names(self) -> dict[int, str]:
        """dictionary of classifier id to classifier  name"""
        return self._sit_identifier_mapping.classifier_names.copy()

    @property
    def classifier_ids(self) -> dict[str, int]:
        """
        dictionary of classifier name to classifier id
        """
        return self._sit_identifier_mapping.classifier_ids.copy()

    @property
    def classifier_value_ids(self) -> dict[str, dict[str, int]]:
        """
        nested dictionary of classifier name (outer key) to classifier value
        name (inner key) to classifier value id.
        """
        return self._sit_identifier_mapping.classifier_value_ids.copy()

    @property
    def classifier_value_names(self) -> dict[int, str]:
        """
        dictionary of classifier_value_id to classifier_value_name
        """
        return self._sit_identifier_mapping.classifier_value_names.copy()

    def get_parameters_factory(self) -> Callable[[], dict]:

        sit_disturbance_types = self._sit_data.disturbance_types.copy()
        sit_disturbance_types.insert(
            0,
            "default_disturbance_type_id",
            self._mapping.get_default_disturbance_type_id(
                sit_disturbance_types.name
            ),
        )

        return self._defaults.get_parameters_factory(sit_disturbance_types)
