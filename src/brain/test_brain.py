"""Self-check: `uv run pytest`, or `uv run python -m brain.test_brain`."""

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco

from brain import adapter, mlp
from brain.rotation import (
    matrix_to_quat,
    quat_to_matrix,
    quat_to_sixd,
    sixd_to_matrix,
    sixd_to_quat,
)

MODEL = str(Path(__file__).parents[2] / "models" / "SELF_Hexapod" / "scene.xml")


def test_rotation():
    q = jax.random.normal(jax.random.key(0), (64, 4))
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)

    # Round trip preserves the rotation. Sign may flip: q and -q are the same
    # rotation, which is exactly the discontinuity 6D exists to avoid.
    back = sixd_to_quat(quat_to_sixd(q))
    assert jnp.allclose(jnp.abs((q * back).sum(-1)), 1.0, atol=1e-5)
    assert jnp.allclose(quat_to_matrix(q), quat_to_matrix(back), atol=1e-5)

    # Gram-Schmidt must yield a proper rotation even from junk input.
    m = sixd_to_matrix(jax.random.normal(jax.random.key(1), (64, 6)))
    assert jnp.allclose(m @ m.swapaxes(-1, -2), jnp.eye(3), atol=1e-5)
    assert jnp.allclose(jnp.linalg.det(m), 1.0, atol=1e-5)

    # 180-degree turn about x: the branch the naive sqrt(1 + trace) formula fails on.
    flip = jnp.array([0.0, 1.0, 0.0, 0.0])
    assert jnp.allclose(jnp.abs(matrix_to_quat(quat_to_matrix(flip))), flip, atol=1e-6)


def test_brain_drives_model():
    mj_model = mujoco.MjModel.from_xml_path(MODEL)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    assert adapter.observe(mj_data).shape == (adapter.obs_dim(mj_model),)

    limits = adapter.joint_limits(mj_model)
    params = mlp.init(
        jax.random.key(0), [adapter.obs_dim(mj_model), 64, 64, mj_model.nu]
    )
    ctrl = adapter.act(params, mj_data, limits)

    assert ctrl.shape == (mj_model.nu,)
    assert jnp.all(ctrl >= limits[0]) and jnp.all(ctrl <= limits[1])

    # It has to survive being stepped, not just produce numbers. Contacts are
    # sampled over the whole rollout: an untrained policy skitters, so ncon at
    # any single instant is flaky.
    touched = False
    for _ in range(400):
        mj_data.ctrl[:] = adapter.act(params, mj_data, limits)
        mujoco.mj_step(mj_model, mj_data)
        touched |= mj_data.ncon > 0
    assert jnp.isfinite(mj_data.qpos).all(), "simulation blew up"
    assert touched, "legs never collided with the floor"
    assert mj_data.qpos[2] > 0, "torso fell through the floor"


if __name__ == "__main__":
    test_rotation()
    test_brain_drives_model()
    print("ok")
