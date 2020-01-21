# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os


def get_script_dir():
    """Returns the directory containing this script file

    Returns:
        str: the directory containing this module
    """
    return os.path.dirname(os.path.realpath(__file__))


def get_query(query_filename):
    """read the contents of a file stored in the same directory as this module.

    Args:
        query_filename (str): the name of a file stored in the same
            directory as this module

    Returns:
        str: the contents of the file
    """
    query_path = os.path.join(get_script_dir(), query_filename)
    with open(query_path, 'r') as query_file:
        return query_file.read()
