import os


def get_script_dir():
    """returns the directory containing this script file
    """
    return os.path.dirname(os.path.realpath(__file__))


def get_query(query_filename):
    """read the contents of a file stored in the same directory as this module.

    Arguments:
        query_filename {str} -- the name of a file stored in the same
            directory as this module

    Returns:
        str -- the contents of the file
    """
    query_path = os.path.join(get_script_dir(), query_filename)
    with open(query_path, 'r') as query_file:
        return query_file.read()
