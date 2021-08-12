import unittest
from mock import Mock, patch
import numpy as np
import pandas as pd
import scipy.sparse
from libcbm.model.moss_c.model_functions import SpinupState
from libcbm.model.moss_c import model_functions
from libcbm.model.moss_c.pools import Pool


class ModelFunctionsTest(unittest.TestCase):

    def test_np_map(self):
        result = model_functions.np_map(
            a=np.array([1, 2, 1, 2, 3, 5]),
            m={1: 2, 2: 1, 3: 4, 5: 4},
            dtype=int)
        self.assertTrue(
            (result == np.array([2, 1, 2, 1, 4, 4], dtype=int)).all())

    def test_np_map_raises_error_on_missing(self):
        with self.assertRaises(ValueError):
            model_functions.np_map(
                a=np.array([1, 2, 1, 2, 3, 5]),
                m={1: 2, 2: 1, 3: 4},
                dtype=int)

    def test_initialize_dm(self):
        disturbance_matrix_data = pd.DataFrame(
            columns=[
                "disturbance_type_id", "source", "sink", "proportion"],
            data=[
                [1, 1, 1, 0.2],
                [1, 1, 2, 0.2],
                [1, 1, 3, 0.6],
                [1, 2, 4, 0.45],
                [1, 2, 2, 0.55],
                [1, 3, 5, 0.5],
                [1, 3, 7, 0.5],
                [1, 4, 1, 1.0],
                [2, 1, 1, 0.2],
                [2, 1, 2, 0.2],
                [2, 1, 3, 0.6],
                [2, 2, 4, 0.45],
                [2, 2, 2, 0.55],
                [2, 3, 5, 0.5],
                [2, 3, 7, 0.5],
                [2, 4, 1, 1.0],
            ])
        pool_map = {int(x): x.name for x in Pool}
        disturbance_matrix_data.source = \
            disturbance_matrix_data.source.map(pool_map)
        disturbance_matrix_data.sink = \
            disturbance_matrix_data.sink.map(pool_map)
        result = model_functions.initialize_dm(
            disturbance_matrix_data)
        self.assertTrue(result.dm_dist_type_index == {0: 0, 1: 1, 2: 2})

        for k, v in result.dm_dist_type_index.items():
            mat_rows = disturbance_matrix_data[
                disturbance_matrix_data.disturbance_type_id == k]
            expected_matrix = np.identity(
                len(Pool), dtype=float)
            for _, row in mat_rows.iterrows():
                expected_matrix[Pool[row.source], Pool[row.sink]] = \
                    row.proportion
            result_matrix = result.dm_list[v]
            result_coo_mat = scipy.sparse.coo_matrix(
                (result_matrix[:, 2],
                    (result_matrix[:, 0].astype(int),
                     result_matrix[:, 1].astype(int))),
                dtype=float)
            self.assertTrue(
                (expected_matrix == result_coo_mat.toarray()).all())

    def test_to_numpy_namespace(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        numpy_namespace = model_functions.to_numpy_namespace(df)
        self.assertTrue(
            df.equals(pd.DataFrame(
                data={k: v for k, v in numpy_namespace.__dict__.items()})))

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
