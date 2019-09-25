import unittest
import numpy as np
from test.wrapper import pool_flux_helpers


class ComputePoolTests(unittest.TestCase):

    def test_single_pool_row_single_matrix_operation(self):
        pools = np.ones((1, 5))

        mat = np.array(
            [[1, 0.5, 0, 0, 0],
             [0, 1.0, 0, 0, 0],
             [0, 0.0, 1, 0, 0],
             [0, 0.0, 0, 1, 0],
             [0, 0.0, 0, 0, 1]])

        op_indices = np.array([[0]], dtype=np.uintp)
        pools_test = pool_flux_helpers.ComputePools(pools, [[mat]], op_indices)

        # create the expected result using the numpy implementation
        pools_expected = np.matmul(pools, mat)

        self.assertTrue((pools_expected - pools_test).sum() == 0)
        self.assertTrue((pools_expected - pools_test).max() == 0)

    def test_multiple_pool_row_single_matrix_operation(self):
        n_stands = 10
        n_pools = 5
        n_ops = 1
        pools = np.ones((n_stands, n_pools))

        # required to be a square matrix of order n-pools
        mat = np.array(
            [[1, 0.5, 0, 0, 0],
             [0, 1.0, 0, 0, 0],
             [0, 0.0, 1, 0, 0],
             [0, 0.0, 0, 1, 0],
             [0, 0.0, 0, 0, 1]])

        op_indices = np.zeros((n_ops, n_stands), dtype=np.uintp)
        pools_test = pool_flux_helpers.ComputePools(pools, [[mat]], op_indices)

        # create the expected result using the numpy implementation
        pools_expected = np.zeros((10, 5))
        for i in range(n_stands):
            pools_expected[i, :] = np.matmul(pools[i, :], mat)

        self.assertTrue((pools_expected - pools_test).sum() == 0)
        self.assertTrue((pools_expected - pools_test).max() == 0)

    def test_multiple_pool_row_multiple_matrix_operation(self):
        """

        In the code below the following vector matrix multiplication
        operations are happening::

            pools[0,:] = pools[0,:] * mat_0 * mat_1
            pools[1,:] = pools[1,:] * mat_0 * mat_3
            pools[2,:] = pools[2,:] * mat_2 * mat_1
            pools[3,:] = pools[3,:] * mat_2 * mat_3

        """

        n_stands = 4
        n_pools = 5
        n_ops = 2
        pools = np.ones((n_stands, n_pools))

        mat_0 = np.array([
            [1, 0.5, 0, 0, 0],
            [0, 1.0, 0, 0, 0],
            [0, 0.0, 1, 0, 0],
            [0, 0.0, 0, 1, 0],
            [0, 0.0, 0, 0, 1]])

        mat_1 = np.array([
            [1, 1.0, 0, 0, 0],
            [0, 1.0, 0, 0, 0],
            [0, 0.0, 1, 0, 0],
            [0, 0.0, 0, 1, 0],
            [0, 0.0, 0, 0, 1]])

        mat_2 = np.array([
            [1, 0.5, 0, 0, 0],
            [0, 1.0, 0, 0, 0],
            [0, 0.0, 1, 0, 0],
            [0, 0.0, 0, 1, 0],
            [0, 0.0, 0, 0, 1]])

        mat_3 = np.array([
            [1, 1.0, 0, 0, 0],
            [0, 1.0, 0, 0, 0],
            [0, 0.0, 1, 0, 0],
            [0, 0.0, 0, 1, 0],
            [0, 0.0, 0, 0, 1]])

        mats = [
            [mat_0, mat_1],
            [mat_2, mat_3]]

        op_indices = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]], dtype=np.uintp)

        # use the libcbm implementation
        pools_test = pool_flux_helpers.ComputePools(pools, mats, op_indices)

        # create the expected result using the numpy implementation
        pools_expected = np.zeros((n_stands, n_pools))
        pools_working = pools.copy()  # working variable required
        for i in range(n_ops):
            for k in range(n_stands):
                mat = mats[i][op_indices[k, i]]
                pools_working[k, :] = np.matmul(pools_working[k, :], mat)

        pools_expected = pools_working

        self.assertTrue((pools_expected - pools_test).sum() == 0)
        self.assertTrue((pools_expected - pools_test).max() == 0)

    def test_randomized_compute_pools(self):
        """runs libcbm compute pools for:

            - a random number of pool rows (stands) [1,1000)
            - a random number of pools [3, 25) with random values [-5e14, 5e14)
            - a random number of ops [1,20)
            - for each op a random number of matrices, with random stand to op
              index associations

        then compute the equivalent result using the numpy.matmul function,
        and check that the result is within a tolerance using numpy.allclose
        """
        np.random.seed(1)
        n_stands = np.random.randint(1, 1000)
        n_pools = np.random.randint(3, 25)
        n_ops = np.random.randint(1, 20)
        pools = (np.random.rand(n_stands, n_pools) - 0.5) * 1e15

        mats = []
        op_indices = np.zeros((n_stands, n_ops), dtype=np.uintp)

        # for each op create a random number of matrices, and create the
        # op indices to associate specific stands with specific matrices
        for i in range(n_ops):
            n_op_mats = int(np.random.rand(1)[0] * n_stands)
            if n_op_mats == 0:
                n_op_mats = 1
            op_indices[:, i] = np.floor(
                (np.random.rand(n_stands) * n_op_mats)).astype(np.uintp)
            op_mats = []
            for _ in range(n_op_mats):
                # create a random square matrix
                op_mats.append(np.random.rand(n_pools, n_pools)*1e5)
            mats.append(op_mats)

        pools_test = pool_flux_helpers.ComputePools(pools, mats, op_indices)

        # working variable required so the original value isn't overwritten
        pools_working = pools.copy()
        # create the expected result using the numpy implementation
        for i in range(n_ops):
            for k in range(n_stands):
                mat = mats[i][op_indices[k, i]]
                pools_working[k, :] = np.matmul(pools_working[k, :], mat)

        pools_expected = pools_working
        self.assertTrue(
            np.allclose(pools_expected, pools_test, rtol=1e-12, atol=1e-15))
