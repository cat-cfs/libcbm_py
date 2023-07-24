# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations
import os
from typing import Callable
import sqlite3
import pandas as pd
from libcbm.resources import cbm_defaults_queries


def load_cbm_parameters(sqlite_path: str) -> dict[str, pd.DataFrame]:
    """Loads cbm default parameters into configuration dictionary format.
    Used for initializing CBM functionality in LibCBM via the InitializeCBM
    function.

    Args:
        sqlite_path (str): Path to a CBM parameters database as formatted
            like: https://github.com/cat-cfs/cbm_defaults

    Raises:
        AssertionError:  if the name of any 2 queries is the same, an error is
            raised.

    Returns:
        dict: a dictionary of name/pandas.DataFrame pairs for use with LibCBM
            configuration.
    """
    result = {}

    queries = {
        k: cbm_defaults_queries.get_query("{}.sql".format(k))
        for k in [
            "decay_parameters",
            "slow_mixing_rate",
            "mean_annual_temp",
            "turnover_parameters",
            "disturbance_matrix_values",
            "disturbance_matrix_associations",
            "root_parameter",
            "growth_multipliers",
            "land_classes",
            "disturbance_type_land_type",
            "spatial_units",
            "random_return_interval",
            "spinup_parameter",
            "afforestation_pre_type",
        ]
    }

    if not os.path.exists(sqlite_path):
        # sqlite3.connect does not raise an error on no path
        raise ValueError(
            "specified path does not exist '{0}'".format(sqlite_path)
        )
    conn = sqlite3.connect(sqlite_path)
    try:
        for table, query in queries.items():
            if table in result:
                raise AssertionError(
                    "duplicate table name detected {}".format(table)
                )
            result[table] = pd.read_sql(query, conn)
    finally:
        conn.close()

    return result


def load_cbm_pools(sqlite_path: str) -> list[dict]:
    """Loads cbm pool information from a cbm_defaults database into the
    format expected by the libcbm compiled library.

    Args:
        sqlite_path (str): path to a cbm_defaults database

    Returns:
        list: list of dictionaries describing CBM pools

            For example::

                [
                    {"name": "pool1", "id": 1, "index": 0},
                    {"name": "pool2", "id": 2, "index": 1},
                    ...,
                    {"name": "poolN", "id": N, "index": N-1},
                ]
    """
    result = []
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    try:
        index = 0
        query = cbm_defaults_queries.get_query("pools.sql")
        for row in cursor.execute(query):
            result.append({"name": row[0], "id": row[1], "index": index})
            index += 1
        return result
    finally:
        cursor.close()
        conn.close()


def load_cbm_flux_indicators(sqlite_path: str) -> list[dict]:
    """Loads cbm flux indicator information from a cbm_defaults database
    into the format expected by the libcbm compiled library.

    Used to capture flows between specified source pools and specified sink
    pools for a given process to return as model output.

    Args:
        sqlite_path (str): path to a cbm_defaults database

    Returns:
        list: list of dictionaries describing CBM flux indicators.

            For example::

                [
                    {
                        "name": "flux_1"
                        "id": 1,
                        "index": 0,
                        "process_id": 1,
                        "source_pools": [1, 2, 3, 4],
                        "sink_pools": [5, 6, 7, 8],
                    },
                ]
    """
    result = []
    flux_indicator_source_sql = cbm_defaults_queries.get_query(
        "flux_indicator_source.sql"
    )
    flux_indicator_sink_sql = cbm_defaults_queries.get_query(
        "flux_indicator_sink.sql"
    )
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    try:
        index = 0
        flux_indicator_sql = cbm_defaults_queries.get_query(
            "flux_indicator.sql"
        )
        flux_indicator_rows = list(cursor.execute(flux_indicator_sql))
        for row in flux_indicator_rows:
            flux_indicator = {
                "id": row[0],
                "name": row[1],
                "index": index,
                "process_id": row[2],
                "source_pools": [],
                "sink_pools": [],
            }
            for source_pool_row in cursor.execute(
                flux_indicator_source_sql, (row[0],)
            ):
                flux_indicator["source_pools"].append(int(source_pool_row[0]))
            for sink_pool_row in cursor.execute(
                flux_indicator_sink_sql, (row[0],)
            ):
                flux_indicator["sink_pools"].append(int(sink_pool_row[0]))
            result.append(flux_indicator)
            index += 1
        return result
    finally:
        cursor.close()
        conn.commit()


def get_cbm_parameters_factory(
    db_path: str,
) -> Callable[[], dict[str, pd.DataFrame]]:
    """Get a function that formats CBM parameters for
    :py:class:`libcbm.wrapper.cbm.cbm_wrapper.CBMWrapper`
    drawn from the specified database.

    Args:
        db_path (str): path to a cbm_defaults database

    Returns:
        func: a function that creates CBM parameters

        Compatible with: :py:func:`libcbm.model.cbm.cbm_factory.create`
    """

    def factory():
        return load_cbm_parameters(db_path)

    return factory


def get_libcbm_configuration_factory(
    db_path: str,
) -> Callable[[], dict[str, list[dict]]]:
    """Get a parameterless function that creates configuration for
    :py:class:`libcbm.wrapper.libcbm_wrapper.LibCBMWrapper`

    Args:
        db_path (str): path to a cbm_defaults database

    Returns:
        func: a function that creates CBM configuration input for libcbm

        Compatible with: :py:func:`libcbm.model.cbm.cbm_factory.create`
    """

    def factory():
        return {
            "pools": load_cbm_pools(db_path),
            "flux_indicators": load_cbm_flux_indicators(db_path),
        }

    return factory
