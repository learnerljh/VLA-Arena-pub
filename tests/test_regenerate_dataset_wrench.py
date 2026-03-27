import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


pytest.importorskip('robosuite')

from scripts.regenerate_dataset import extract_ee_wrench


def _add_rlds_builder_to_path():
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[1] / 'rlds_dataset_builder'),
    )


def _make_env(sensor_names=(), sensor_values=()):
    model = SimpleNamespace(
        nsensor=len(sensor_names),
        sensor_adr=np.arange(0, len(sensor_names) * 3, 3, dtype=np.int32),
        sensor_dim=np.full(len(sensor_names), 3, dtype=np.int32),
    )

    def sensor_id2name(sensor_id):
        return sensor_names[sensor_id]

    model.sensor_id2name = sensor_id2name
    data = SimpleNamespace(sensordata=np.asarray(sensor_values, dtype=np.float32))
    sim = SimpleNamespace(model=model, data=data)
    return SimpleNamespace(sim=sim)


def test_extract_ee_wrench_prefers_observation_keys():
    env = _make_env()
    obs = {
        'robot0_eef_force': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'robot0_eef_torque': np.array([4.0, 5.0, 6.0], dtype=np.float32),
    }

    force, torque, source, _ = extract_ee_wrench(env, obs)

    np.testing.assert_allclose(force, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(torque, [4.0, 5.0, 6.0])
    assert source == (
        'force=obs:robot0_eef_force;torque=obs:robot0_eef_torque'
    )


def test_extract_ee_wrench_falls_back_to_sensor_names():
    env = _make_env(
        sensor_names=('robot0_wrist_force', 'robot0_wrist_torque'),
        sensor_values=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    )

    force, torque, source, _ = extract_ee_wrench(env, {})

    np.testing.assert_allclose(force, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(torque, [0.4, 0.5, 0.6])
    assert source == (
        'force=sensor:robot0_wrist_force;torque=sensor:robot0_wrist_torque'
    )


def test_load_ee_force_and_torque_backfills_old_hdf5(tmp_path):
    pytest.importorskip('tensorflow_datasets')
    _add_rlds_builder_to_path()
    from VLA_Arena.VLA_Arena_dataset_builder import _load_ee_force_and_torque

    file_path = tmp_path / 'demo.hdf5'
    with h5py.File(file_path, 'w') as h5_file:
        obs = h5_file.create_group('obs')
        force, torque = _load_ee_force_and_torque(obs, 2)

    np.testing.assert_allclose(force, np.zeros((2, 3), dtype=np.float32))
    np.testing.assert_allclose(torque, np.zeros((2, 3), dtype=np.float32))


def test_load_ee_force_and_torque_splits_wrench_dataset(tmp_path):
    pytest.importorskip('tensorflow_datasets')
    _add_rlds_builder_to_path()
    from VLA_Arena.VLA_Arena_dataset_builder import _load_ee_force_and_torque

    file_path = tmp_path / 'demo.hdf5'
    with h5py.File(file_path, 'w') as h5_file:
        obs = h5_file.create_group('obs')
        obs.create_dataset(
            'ee_wrench',
            data=np.array(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                ],
                dtype=np.float32,
            ),
        )
        force, torque = _load_ee_force_and_torque(obs, 2)

    np.testing.assert_allclose(
        force,
        np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        torque,
        np.array([[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]], dtype=np.float32),
    )
