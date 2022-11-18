import numpy as np
from libcbm.storage import series
from libcbm.model.model_definition.spinup_engine import SpinupState
from libcbm.model.model_definition import spinup_engine


def test_advance_spinup_state():
    def run_test(expected_output, **input_kwargs):
        test_kwargs = {
            k: series.from_numpy(k, np.array([v]))
            for k, v in input_kwargs.items()
        }
        out = spinup_engine.advance_spinup_state(**test_kwargs)
        assert expected_output == out

    run_test(
        expected_output=SpinupState.AnnualProcesses,
        spinup_state=SpinupState.AnnualProcesses,
        age=0,
        final_age=100,
        return_interval=100,
        rotation_num=0,
        max_rotations=10,
        last_rotation_slow=0,
        this_rotation_slow=0,
    )

    run_test(
        expected_output=SpinupState.HistoricalEvent,
        spinup_state=SpinupState.AnnualProcesses,
        age=100,
        final_age=100,
        return_interval=100,
        rotation_num=0,
        max_rotations=10,
        last_rotation_slow=50,
        this_rotation_slow=100,
    )

    run_test(
        expected_output=SpinupState.LastPassEvent,
        spinup_state=SpinupState.AnnualProcesses,
        age=100,
        final_age=100,
        return_interval=100,
        rotation_num=7,
        max_rotations=10,
        last_rotation_slow=99.9999,
        this_rotation_slow=100,
    )

    run_test(
        expected_output=SpinupState.LastPassEvent,
        spinup_state=SpinupState.AnnualProcesses,
        age=100,
        final_age=100,
        return_interval=100,
        rotation_num=10,
        max_rotations=10,
        last_rotation_slow=60,
        this_rotation_slow=100,
    )

    run_test(
        expected_output=SpinupState.AnnualProcesses,
        spinup_state=SpinupState.HistoricalEvent,
        age=100,
        final_age=100,
        return_interval=100,
        rotation_num=9,
        max_rotations=10,
        last_rotation_slow=60,
        this_rotation_slow=100,
    )

    run_test(
        expected_output=SpinupState.GrowToFinalAge,
        spinup_state=SpinupState.LastPassEvent,
        age=75,
        final_age=100,
        return_interval=100,
        rotation_num=10,
        max_rotations=10,
        last_rotation_slow=60,
        this_rotation_slow=100,
    )

    run_test(
        expected_output=SpinupState.End,
        spinup_state=SpinupState.LastPassEvent,
        age=0,
        final_age=0,
        return_interval=100,
        rotation_num=6,
        max_rotations=10,
        last_rotation_slow=99.9999,
        this_rotation_slow=100,
    )

    run_test(
        expected_output=SpinupState.GrowToFinalAge,
        spinup_state=SpinupState.GrowToFinalAge,
        age=0,
        final_age=10,
        return_interval=100,
        rotation_num=10,
        max_rotations=10,
        last_rotation_slow=60,
        this_rotation_slow=100,
    )

    run_test(
        expected_output=SpinupState.End,
        spinup_state=SpinupState.GrowToFinalAge,
        age=10,
        final_age=10,
        return_interval=100,
        rotation_num=10,
        max_rotations=10,
        last_rotation_slow=60,
        this_rotation_slow=100,
    )
