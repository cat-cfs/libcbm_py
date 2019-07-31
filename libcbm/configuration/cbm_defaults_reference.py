# methods for finding name/id associations in a CBM defaults
# database

import sqlite3
import pandas as pd
import libcbm.configuration.cbm_defaults_queries as queries

# queries for species name/species id associations
species_reference_query = queries.get_query("species_name_ref.sql")

# queries for spatial unit id, admin boundary name, eco boundary name
# associations
spatial_unit_reference_query = queries.get_query("spatial_units_name_ref.sql")

# queries information on disturbance types which have an effect on UNFCCC land
# class
land_class_disturbance_query = queries.get_query(
    "land_class_disturbance_ref.sql")

# queries for land class name,id,code,descriptions
land_class_query = queries.get_query("land_class_ref.sql")

# queries for afforestation pre-type/name associations
afforestation_pre_type_ref = queries.get_query(
    "afforestation_pre_type_ref.sql")

# queries for names of flux indicators id/name associations
flux_indicator_ref = queries.get_query("flux_indicator_ref.sql")


def load_data(sqlite_path, query, locale_code="en-CA", as_data_frame=False):
    """loads the specified query into a list of dictionary formatted query

    Arguments:
        sqlite_path {str} -- path to a SQLite database
        query {str}  -- sqlite query

    Keyword Arguments:
        locale_code {str} -- [description] (default: {"en-CA"})

    Returns:
        [type] -- [description]
    """
    if not as_data_frame:
        result = []
        with sqlite3.connect(sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            for row in cursor.execute(query, (locale_code,)):
                result.append(row)
        return result
    else:
        with sqlite3.connect(sqlite_path) as conn:
            df = pd.read_sql_query(sql=query, con=conn, params=(locale_code,))
            return df
