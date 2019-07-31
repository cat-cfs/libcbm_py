# methods for finding name/id associations in a CBM defaults
# database

import sqlite3
import pandas as pd
import libcbm.configuration.cbm_defaults_queries as queries

# queries for species name/species id associations
species_reference_query = queries.get_query("species_name_ref.sql")

disturbance_reference_query = queries.get_query("disturbance_type_ref.sql")

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
afforestation_pre_type_query = queries.get_query(
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


class CBMDefaultsReference:

    def __init__(self, sqlite_path, locale_code="en-CA"):

        self.species_ref = load_data(
            sqlite_path, species_reference_query, locale_code)
        self.species_by_name = {x["name"]: x for x in self.species_ref}

        self.disturbance_type_ref = load_data(
            sqlite_path, disturbance_reference_query, locale_code)
        self.disturbance_type_by_name = {
            x["name"]: x for x in self.disturbance_type_ref}

        self.spatial_unit_ref = load_data(
            sqlite_path, spatial_unit_reference_query, locale_code)
        self.spatial_unit_by_admin_eco_names = {
            (x["admin_boundary_name"], x["eco_boundary_name"]): x
            for x in self.spatial_unit_ref}

        self.afforestation_pre_type_ref = load_data(
            sqlite_path, afforestation_pre_type_query, locale_code)
        self.afforestation_pre_type_by_name = {
            x["name"]: x for x in self.afforestation_pre_type_ref}

        self.land_class_ref = load_data(
            sqlite_path, land_class_query, locale_code)
        self.land_class_by_code = {
            x["code"]: x for x in self.land_class_ref}

    def get_species_id(self, species_name):
        """Get the species id associated with the specified species name.

        Arguments:
            species_name {str} -- a species name

        Returns:
            int -- the id associated with the specified name
        """
        return self.species_by_name[species_name]["id"]

    def get_disturbance_type_id(self, disturbance_type_name):
        """Get the disturbance type id associated with the specified
        disturbance type name

        Arguments:
            disturbance_type_name {str} -- a disturbance type name

        Returns:
            int -- disturbance type id
        """
        return self.disturbance_type_by_name[disturbance_type_name]["id"]

    def get_spatial_unit_id(self, admin_boundary_name, eco_boundary_name):
        """Get the spatial unit id associated with the specified
        admin-boundary-name, eco-boundary-name combination

        Arguments:
            admin_boundary_name {str} -- an admin boundary name
            eco_boundary_name {[type]} -- an eco boundary name

        Returns:
            int -- the spatial unit id
        """
        return self.spatial_unit_by_admin_eco_names[
            (admin_boundary_name, eco_boundary_name)]["id"]

    def get_afforestation_pre_type_id(self, afforestation_pre_type_name):
        """Get the afforestation pre-type id associated with the specified
        afforestation pre-type name

        Arguments:
            afforestation_pre_type_name {str} -- [description]

        Returns:
            [type] -- [description]
        """
        return self.afforestation_pre_type_by_name[
            afforestation_pre_type_name]["id"]

    def get_land_class_id(self, land_class_code):
        """Get the land class id associated with the specified CBM land class
        code (where a code might be for example: UNFCCC_FL_R_FL)

        Arguments:
            land_class_code {str} -- a CBM formatted UNFCCC land class code

        Returns:
            int -- the land class id associated with the code
        """
        return self.land_class_by_code[land_class_code]["id"]