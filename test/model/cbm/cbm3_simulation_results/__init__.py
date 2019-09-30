import os
from libcbm.test.cbm.cbm3_support import cbm3_test_io


def get_local_dir():
    return os.path.dirname(os.path.realpath(__file__))


def get_results(sub_directory):
    """loads a saved CBM3 simulation result given the sub directory
    containing results.  The contents of the directory are the same
    as the result of running:
    :py:func:`libcbm.test.cbm.cbm3_support.cbm3_test_io.save_cbm_cfs3_test`

    Args:
        directory (sub_directory): a subdirectory in the same path as this
            module containing CBM3 results

    Returns:
        types.SimpleNamespace: the return value of:
            :py:func:`libcbm.test.cbm.cbm3_support.cbm3_test_io.load_cbm_cfs3_test`
    """
    cbm3_result = cbm3_test_io.load_cbm_cfs3_test(
        os.path.join(get_local_dir(), sub_directory))
    return cbm3_result
