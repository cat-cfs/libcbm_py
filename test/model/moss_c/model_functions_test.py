import unittest
from mock import Mock, patch
import numpy as np
from libcbm.model.moss_c.model_functions import SpinupState
from libcbm.model.moss_c import model_functions


class ModelFunctionsTest(unittest.TestCase):

    def test_expand_matrix(self):
        result = model_functions.expand_matrix(mat=[
            [1,  2,   np.array([3,   4,  5])],
            [6,  7,   np.array([8,   9, 10])],
            [11, 12,  np.array([13, 14, 15])],
        ], identity_set={})

        expected_output = [
            np.array([
                [1, 2, 3],
                [6, 7, 8],
                [11, 12, 13]], dtype=float),
            np.array([
                [1, 2, 4],
                [6, 7, 9],
                [11, 12, 14]], dtype=float),
            np.array([
                [1, 2, 5],
                [6, 7, 10],
                [11, 12, 15]], dtype=float),
        ]
        for i_mat, mat in enumerate(result):
            self.assertTrue((mat == expected_output[i_mat]).all())

    def test_expand_matrix_identity_set(self):
        result = model_functions.expand_matrix(mat=[
            [1,  1,   np.array([3,   4,  5])],
            [6,  7,   np.array([8,   9, 10])],
            [11, 12,  np.array([13, 14, 15])],
        ], identity_set={1, 2, 3})

        expected_output = [
            np.array([
                [2, 2, 1],
                [3, 3, 1],
                [1, 1, 3],
                [6, 7, 8],
                [11, 12, 13]], dtype=float),
            np.array([
                [2, 2, 1],
                [3, 3, 1],
                [1, 1, 4],
                [6, 7, 9],
                [11, 12, 14]], dtype=float),
            np.array([
                [2, 2, 1],
                [3, 3, 1],
                [1, 1, 5],
                [6, 7, 10],
                [11, 12, 15]], dtype=float),
        ]
        for i_mat, mat in enumerate(result):
            self.assertTrue((mat == expected_output[i_mat]).all())


    def test_compute_with_pools_only(self):
        dll=Mock()
        model_functions.compute(
            dll=dll,
            pools="pools",
            ops="ops",
            op_indices="op_indices",
            op_processes=None,
            flux=None,
            enabled="enabled")


    def test_advance_spinup_state(self):

        def run_test(expected_output, **input_kwargs):
            test_kwargs = {
                k: np.array([v])
                for k, v in input_kwargs.items()
            }
            out = model_functions.advance_spinup_state(**test_kwargs)
            self.assertTrue(expected_output == out)

        run_test(
            expected_output=SpinupState.AnnualProcesses,
            spinup_state=SpinupState.AnnualProcesses,
            age=0,
            final_age=100,
            return_interval=100,
            rotation_num=0,
            max_rotations=10,
            last_rotation_slow=0,
            this_rotation_slow=0)

        run_test(
            expected_output=SpinupState.HistoricalEvent,
            spinup_state=SpinupState.AnnualProcesses,
            age=100,
            final_age=100,
            return_interval=100,
            rotation_num=0,
            max_rotations=10,
            last_rotation_slow=50,
            this_rotation_slow=100)

        run_test(
            expected_output=SpinupState.LastPassEvent,
            spinup_state=SpinupState.AnnualProcesses,
            age=100,
            final_age=100,
            return_interval=100,
            rotation_num=7,
            max_rotations=10,
            last_rotation_slow=99.9999,
            this_rotation_slow=100)

        run_test(
            expected_output=SpinupState.LastPassEvent,
            spinup_state=SpinupState.AnnualProcesses,
            age=100,
            final_age=100,
            return_interval=100,
            rotation_num=10,
            max_rotations=10,
            last_rotation_slow=60,
            this_rotation_slow=100)

        run_test(
            expected_output=SpinupState.AnnualProcesses,
            spinup_state=SpinupState.HistoricalEvent,
            age=100,
            final_age=100,
            return_interval=100,
            rotation_num=9,
            max_rotations=10,
            last_rotation_slow=60,
            this_rotation_slow=100)

        run_test(
            expected_output=SpinupState.GrowToFinalAge,
            spinup_state=SpinupState.LastPassEvent,
            age=75,
            final_age=100,
            return_interval=100,
            rotation_num=10,
            max_rotations=10,
            last_rotation_slow=60,
            this_rotation_slow=100)

        run_test(
            expected_output=SpinupState.End,
            spinup_state=SpinupState.LastPassEvent,
            age=0,
            final_age=0,
            return_interval=100,
            rotation_num=6,
            max_rotations=10,
            last_rotation_slow=99.9999,
            this_rotation_slow=100)

        run_test(
            expected_output=SpinupState.GrowToFinalAge,
            spinup_state=SpinupState.GrowToFinalAge,
            age=0,
            final_age=10,
            return_interval=100,
            rotation_num=10,
            max_rotations=10,
            last_rotation_slow=60,
            this_rotation_slow=100)

        run_test(
            expected_output=SpinupState.End,
            spinup_state=SpinupState.GrowToFinalAge,
            age=10,
            final_age=10,
            return_interval=100,
            rotation_num=10,
            max_rotations=10,
            last_rotation_slow=60,
            this_rotation_slow=100)
