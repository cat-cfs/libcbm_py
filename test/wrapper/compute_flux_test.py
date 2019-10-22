import unittest
import numpy as np
from test.wrapper import pool_flux_helpers


class ComputeFluxTests(unittest.TestCase):

    def test_compute_flux(self):
        """Tests the libcbm ComputeFlux method with random inputs
        """

        np.random.seed(1)
        flux_indicator_config = [
            # with this flux indicator, we are capturing all flows from the "a"
            # pool to any of the other pools
            {"process_id": 1, "sinks": ["b", "c", "d", "e"], "sources": ["a"]},
            # and with this one, we are capturing all flows to the "a" pool
            # from any of the other pools
            {"process_id": 2, "sinks": ["a"], "sources": ["b", "c", "d", "e"]},
            {"process_id": 2, "sinks": ["a"], "sources": ["a"]},
            {"process_id": 3, "sinks": ["d", "e"], "sources": ["b", "c"]},
        ]
        unique_process_ids = {x["process_id"] for x in flux_indicator_config}
        poolnames = ["a", "b", "c", "d", "e"]
        pool_index = {x: i for i, x in enumerate(poolnames)}
        n_stands = np.random.randint(1, 1000)
        n_pools = len(poolnames)
        n_ops = np.random.randint(1, 20)
        pools = (np.random.rand(n_stands, n_pools)-0.5)*0.1

        mats = []
        op_indices = np.zeros((n_stands, n_ops), dtype=np.uintp)
        for i in range(n_ops):
            n_op_mats = int(np.random.rand(1)[0] * n_stands)
            if n_op_mats == 0:
                n_op_mats = 1
            op_indices[:, i] = np.floor(
                (np.random.rand(n_stands) * n_op_mats)).astype(np.uintp)
            op_mats = []
            for _ in range(n_op_mats):
                # create a random square matrix
                op_mats.append(np.random.rand(n_pools, n_pools))
            mats.append(op_mats)

        # evenly assigns ops to the defined process ids
        op_processes = [x % len(unique_process_ids)+1 for x in range(n_ops)]
        pools_test, flux_test = pool_flux_helpers.ComputeFlux(
            pools, poolnames, mats, op_indices, op_processes,
            flux_indicator_config)

        # create the expected result using the numpy implementation
        # this fully emulates the ComputeFlux function, and computes an
        # independent result against which we check differences
        identity = np.identity(n_pools)
        pools_working = pools.copy()  # working variable required
        flux_expected = np.zeros((n_stands, len(flux_indicator_config)))
        for i in range(n_ops):
            for k in range(n_stands):
                mat = mats[i][op_indices[k, i]]
                flux = np.matmul(
                    np.diag(pools_working[k, :]), (mat - identity))
                for i_f, f_config in enumerate(flux_indicator_config):
                    process_id = op_processes[i]
                    if f_config["process_id"] != process_id:
                        continue
                    for src in f_config["sources"]:
                        for sink in f_config["sinks"]:
                            flux_expected[k, i_f] += flux[
                                pool_index[src], pool_index[sink]]
                pools_working[k, :] = np.matmul(pools_working[k, :], mat)

        pools_expected = pools_working

        self.assertTrue(
            np.allclose(pools_expected, pools_test, rtol=1e-12, atol=1e-15))

        self.assertTrue(
            np.allclose(flux_expected, flux_test, rtol=1e-12, atol=1e-15))
