import numpy as np
import pytest

from orca_sim import (
    OrcaHandCombined,
    OrcaHandLeft,
    OrcaHandRight,
    OrcaHandRightCubeOrientation,
)
from orca_sim.envs import _disable_skin_geoms


@pytest.mark.parametrize(
    ("env_cls", "obs_size", "action_size", "version"),
    [
        (OrcaHandLeft, 34, 17, "v1"),
        (OrcaHandLeft, 34, 17, "v2"),
        (OrcaHandRight, 34, 17, "v1"),
        (OrcaHandRight, 34, 17, "v2"),
        (OrcaHandCombined, 68, 34, "v1"),
        (OrcaHandCombined, 68, 34, "v2"),
    ],
)
def test_env_reset_and_step_smoke(
    env_cls, obs_size: int, action_size: int, version: str
) -> None:
    env = env_cls(version=version)
    try:
        obs, info = env.reset()

        assert obs.shape == (obs_size,)
        assert info == {}
        assert env.action_space.shape == (action_size,)

        next_obs, reward, terminated, truncated, next_info = env.step(
            env.action_space.sample()
        )

        assert next_obs.shape == (obs_size,), "Next observation shape is not correct"
        assert isinstance(reward, float), "Reward is not a float"
        assert isinstance(terminated, bool), "Terminated is not a bool"
        assert isinstance(truncated, bool), "Truncated is not a bool"
        assert isinstance(next_info, dict), "Next info is not a dict"
    finally:
        env.close()


def test_reset_accepts_explicit_qpos_and_qvel() -> None:
    env = OrcaHandRight()
    try:
        qpos = np.linspace(-0.1, 0.1, env.data.qpos.size)
        qvel = np.linspace(-0.2, 0.2, env.data.qvel.size)

        obs, _ = env.reset(options={"qpos": qpos, "qvel": qvel})

        np.testing.assert_allclose(env.data.qpos, qpos)
        np.testing.assert_allclose(env.data.qvel, qvel)
        np.testing.assert_allclose(obs[: env.data.qpos.size], qpos)
        np.testing.assert_allclose(obs[env.data.qpos.size :], qvel)
    finally:
        env.close()


def test_step_clips_actions_to_actuator_limits() -> None:
    env = OrcaHandLeft()
    try:
        env.reset()
        env.step(env.action_high + 10.0)
        np.testing.assert_allclose(env.data.ctrl, env.action_high)
    finally:
        env.close()


def test_invalid_render_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported render_mode"):
        OrcaHandRight(render_mode="wireframe")


def test_reset_rejects_wrong_qpos_shape() -> None:
    env = OrcaHandRight()
    try:
        wrong_shape = np.zeros(env.data.qpos.size + 1)
        with pytest.raises(ValueError, match="Expected qpos shape"):
            env.reset(options={"qpos": wrong_shape})
    finally:
        env.close()


def test_step_rejects_wrong_action_shape() -> None:
    env = OrcaHandRight()
    try:
        env.reset()
        wrong_shape = np.zeros(env.action_space.shape[0] + 1, dtype=np.float32)
        with pytest.raises(ValueError, match="Expected action shape"):
            env.step(wrong_shape)
    finally:
        env.close()


def _count_skin_geoms(model) -> tuple[int, int, int]:
    """Return (total, visible, colliding) counts for skin mesh geoms."""
    total = visible = colliding = 0
    for i in range(model.ngeom):
        if model.geom_type[i] != 7:  # mjGEOM_MESH
            continue
        mid = int(model.geom_dataid[i])
        if mid < 0 or "skin" not in (model.mesh(mid).name or "").lower():
            continue
        total += 1
        if model.geom_rgba[i, 3] > 0.01:
            visible += 1
        if model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0:
            colliding += 1
    return total, visible, colliding


@pytest.mark.parametrize(
    ("env_cls", "obs_size", "action_size", "version"),
    [
        (OrcaHandRight, 34, 17, "v1"),
        (OrcaHandLeft, 34, 17, "v1"),
        (OrcaHandRightCubeOrientation, 51, 17, "v1"),
        (OrcaHandRight, 34, 17, "v2"),
    ],
)
def test_skin_false_disables_skin_geoms(env_cls, obs_size, action_size, version) -> None:
    """skin=True (default) keeps skin geoms visible/colliding as authored;
    skin=False makes every *_skin mesh geom invisible and non-colliding."""
    env_with = env_cls(version=version, skin=True)
    env_without = env_cls(version=version, skin=False)
    try:
        env_with.reset()
        env_without.reset()

        total_with, vis_with, col_with = _count_skin_geoms(env_with.model)
        total_without, vis_without, col_without = _count_skin_geoms(env_without.model)

        assert total_with > 0, "Test premise: env should ship with at least one skin geom"
        assert total_without == total_with, "skin=False should not remove geoms from the model"
        assert vis_with > 0, "Test premise: skin=True should have visible skin"
        assert vis_without == 0, "skin=False must zero out skin geom alpha"
        # v1 ships skin geoms with non-zero collision; v2 keeps them visual-only.
        # Either way, skin=False must collapse any collision contribution to zero.
        if col_with > 0:
            assert col_without == 0, "skin=False must disable skin geom collision (contype=conaffinity=0)"
        assert env_without._disabled_skin_geom_count == total_with
        assert env_with._disabled_skin_geom_count == 0

        # Shapes and obs/action spaces are unchanged.
        assert env_without.observation_space.shape == (obs_size,)
        assert env_without.action_space.shape == (action_size,)
    finally:
        env_with.close()
        env_without.close()


