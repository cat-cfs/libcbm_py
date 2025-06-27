# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations


def classifier_value(value: str, description: str = "") -> dict[str, str]:
    """Composes a classifier value structure used for configuring libcbm.

    Args:
        value (str): the classifier value string identifier
        description (str, optional): An optional classifier value description.
            Defaults to "".

    Returns:
        dict: a dictionary formatted for libcbm.

            Example return value::

                {
                    "value": "c1_v1",
                    "description": "c1_v1"
                }
    """
    return {"value": value, "description": description}


def classifier(name: str, values: dict) -> dict:
    """Composes a classifier structure used for configuring libcbm.

    Args:
        name (str): the name of the classifier
        values (dict): The classifier values that compose this classifier.
            Use classifier values of the format returned by:
            :py:func:`classifier_value`

    Returns:
        dict: A dictionary with the following keys:

            - "classifier": The classifier
            - "classifier_values": The list of classifier value dictionaries
                associated with the classifier

            Example return value::

                {
                    "classifier": {"name": "c1"},
                    "classifier_values": [
                        {
                            "value": "c1_v1",
                            "description": "c1_v1"
                        },
                        ...
                    ]
                }
    """
    return {"classifier": {"name": name}, "classifier_values": values}


def classifier_config(classifiers: list) -> dict:
    """Compose the classifier value scheme used for configuring libcbm.
    This method will generate classifier ids and classifier value ids
    based on the order of classifiers and classifier values.

    Args:
        classifiers (list): A list of dictionaries describing classifiers.
            For example::

                [
                    {
                        "classifier": {
                            "name" "c1"
                        },
                        "classifier_values":[
                            {"value": "c1_v1", "description": "c1_v1"},
                            {"value": "c1_v2", "description": "c1_v2"},
                            ...
                            {"value": "c1_vn", "description": "c1_vn"}
                        ]
                    },
                    {
                        "classifier": {
                            "name" "c2"
                        },
                        "classifier_values":[
                            {"value": "c2_v1", "description": "c2_v1"},
                            {"value": "c2_v2", "description": "c2_v2"},
                            ...
                            {"value": "c2_vk", "description": "c1_vk"}
                        ]
                    },
                    ...,
                    {
                        "classifier": {
                            "name" "cj"
                        },
                        "classifier_values":[
                            {"value": "cj_v1", "description": "cj_v1"},
                            {"value": "cj_v2", "description": "cj_v2"},
                            ...
                            {"value": "cj_vi", "description": "cj_vi"}
                        ]
                    }
                ]

            where:
            - j is the number of classifiers,
            - n,k,i are the number of classifier values for each
            corresponding classifier

            The dictionaries composing this argument can be created by the
            :py:func:`classifier_value`, and :py:func:`classifier` functions

    Returns:
        dict: A dictionary with the following keys

            - "classifiers": the list of classifiers
            - "classifier_values": the list of classifier values

            This is the format required by the libcbm CBM model component.

            For example: (following the same pattern given in the input)::

                {
                "classifiers": [
                    {"id": 1, "name": "c1"},
                    {"id": 2, "name": "c2"},
                    ...,
                    {"id": j, "name": "cj"}
                ],
                "classifier_values": [
                    {"id": 1, "classifier_id": 1, "value": "c1_v1", ...},
                    {"id": 2, "classifier_id": 1, "value": "c1_v2", ...},
                    ...,
                    {"id": n, "classifier_id": 1, "value": "c1_vn", ...},
                    {"id": n+1, "classifier_id": 2, "value": "c2_v1", ...},
                    {"id": n+2, "classifier_id": 2, "value": "c2_v2", ...},
                    ...,
                    {"id": n+k, "classifier_id": 2, "value": "c2_vk", ...},
                    {"id": n+k+1, "classifier_id": j, "value": "cj_v1", ...},
                    {"id": n+k+2, "classifier_id": j, "value": "cj_v2", ...},
                    ...,
                    {"id": n+k+i, "classifier_id": j, "value": "cj_vi", ...}
                ]
                }
    """
    result = {
        "classifiers": [],
        "classifier_values": [],
    }
    for i, c in enumerate(classifiers):
        _classifier = c["classifier"]
        values = c["classifier_values"]
        _classifier["id"] = i + 1
        result["classifiers"].append(_classifier)
        for value in values:
            value["id"] = len(result["classifier_values"]) + 1
            value["classifier_id"] = _classifier["id"]
            result["classifier_values"].append(value)
    return result


def get_classifier_indexes(classifier_config: dict) -> dict:
    """Build an object with indexes for the specified classifier_config.

    Args:
        classifier_config (dict):  See the return value of
            :py:func:`classifier_config` for the format.

    Returns:
        dict: a dictionary with the following members:

            * "classifier_names": dictionary of classifier id to classifier
              name
            * "classifier_ids": dictionary of classifier name to classifier
              id
            * "classifier_value_ids": nested dictionary of classifier name
              (outer key) to classifier value name (inner key) to classifier
              value id.
            * "classifier_value_names": dictionary of classifier_value_id to
               classifier_value_name
    """
    indexes = dict(
        classifier_names={},
        classifier_ids={},
        classifier_value_ids={},
        classifier_value_names={},
    )

    for classifier_data in classifier_config["classifiers"]:
        indexes["classifier_names"][classifier_data["id"]] = classifier_data[
            "name"
        ]
        indexes["classifier_ids"][classifier_data["name"]] = classifier_data[
            "id"
        ]

    for classifier_value_data in classifier_config["classifier_values"]:
        classifier_id = classifier_value_data["classifier_id"]
        classifier_name = indexes["classifier_names"][classifier_id]
        classifier_value_id = classifier_value_data["id"]
        classifier_value_name = classifier_value_data["value"]

        if classifier_name in indexes["classifier_value_ids"]:
            indexes["classifier_value_ids"][classifier_name][
                classifier_value_name
            ] = classifier_value_id
        else:
            indexes["classifier_value_ids"][classifier_name] = {
                classifier_value_name: classifier_value_id
            }

        if classifier_value_id not in indexes["classifier_value_names"]:
            indexes["classifier_value_names"][
                classifier_value_id
            ] = classifier_value_name
        else:
            raise ValueError(
                f"classifier_value_id {classifier_value_id} associated with "
                f"more than one classifier value: {classifier_value_name}, "
                f"{indexes['classifier_value_names'][classifier_value_id]}"
            )

    return indexes


def merch_volume_curve(
    classifier_set: list[str], merch_volumes: list[dict]
) -> dict:
    """Formats merchantable volume growth curve data for libcbm CBM model
    consumption.

    Args:
        classifier_set (list): a list of classifier value names ordered by
            classifiers. For example::

                ["c1_v1", "c2_v1", ... "ck_vx"]

        merch_volumes (list): a list of dictionaries with keys:

            - species_id,
            - age_volume_pairs

            for example::

                [
                    {
                        "species_id": 1,
                        "age_volume_pairs": [(a1,v1),(a2,v2),...,(an,vn)]
                    },
                    {
                        "species_id": 6,
                        "age_volume_pairs": [(a1,v1),(a2,v2),...,(an,vn)]
                    },
                ]


    Returns:
        dict: a dictionary with keys `classifier_set` and `components`.

            For example::

                {
                "classifier_set": {
                    "type": "name",
                    "values": ["c1_v1", "c2_v1", ... "ck_vx"]
                },
                "components": [
                    {
                        "species_id": 1,
                        "age_volume_pairs": [(a1,v1),(a2,v2),...,(an,vn)]
                    },
                    {
                        "species_id": 6,
                        "age_volume_pairs": [(a1,v1),(a2,v2),...,(an,vn)]
                    },
                ]
                }

    """
    result = {
        "classifier_set": {
            "type": "name",
            "values": [x for x in classifier_set],
        },
    }
    components = []
    for m in merch_volumes:
        components.append(
            {
                "species_id": m["species_id"],
                "age_volume_pairs": m["age_volume_pairs"],
            }
        )
    result["components"] = components
    return result


def merch_volume_to_biomass_config(
    db_path: str, merch_volume_curves: list, use_smoother: bool = True
) -> dict:
    """Formats merchantable volume growth curve data for libcbm CBM model
    consumption.

    Args:
        db_path (str): path to a cbm_defaults database
        merch_volume_curves (list): a list of dictionaries in the same format
            as the return value of :py:func:`merch_volume_curve`
        use_smoother (bool, optional): use the volume to biomass smoother (on
            by default)

    Returns:
        dict: A dictionary containing configuration for merchantable volume
              growth in the CBM model

              For example::

                {
                    "db_path": "cbm_defaults.db",
                    "use_smoother": True,
                    "merch_volume_curves": [
                        {
                            "classifier_set": {
                                "type": "name",
                                "values": ["c1_v1", "c2_v1", ... "ck_vx"]
                            },
                            "components": [
                                {
                                    "species_id": 1,
                                    "age_volume_pairs": [(a1,v1),
                                                         (a2,v2),
                                                         ...,
                                                         (an,vn)]
                                },
                                {
                                    "species_id": 6,
                                    "age_volume_pairs": [(a1,v1),
                                                         (a2,v2),
                                                         ...,
                                                         (an,vn)]
                                },
                            ]
                        },
                        {
                            "classifier_set": {
                                "type": "name",
                                "values": ["c1_v5", "c2_v1", ... "ck_vx"]
                            },
                            "components": [
                                {
                                    "species_id": 2,
                                    "age_volume_pairs": [(a1,v1),
                                                         (a2,v2),
                                                         ...,
                                                         (an,vn)]
                                },
                                {
                                    "species_id": 65,
                                    "age_volume_pairs": [(a1,v1),
                                                         (a2,v2),
                                                         ...,
                                                         (an,vn)]
                                },
                            ]
                        }
                    ]
                }
    """
    return {"db_path": db_path, "merch_volume_curves": merch_volume_curves,
            "use_smoother": use_smoother,}
