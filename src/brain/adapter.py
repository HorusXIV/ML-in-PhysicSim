"""Glue between the hexapod and the MLP.

Works on either an MjData (numpy) or an mjx.Data (jax) — both expose qpos/qvel
with the same layout, and jnp handles either array type.

Layout for a free-jointed body: qpos = [xyz(3), quat(4), joints...]
                                qvel = [linvel(3), angvel(3), joints...]
"""

import jax.numpy as jnp

from brain import mlp
from brain.rotation import quat_to_sixd


def obs_dim(mj_model):
    """1 height + 6 orientation + 6 root velocity + joint angles + joint velocities."""
    return 13 + (mj_model.nq - 7) + (mj_model.nv - 6)


def observe(data):
    """Everything the brain senses, as one flat vector.

    Absolute x/y are deliberately left out: a gait should not depend on where
    in the world it started. Orientation goes in as 6D rather than the raw
    quaternion so the input is continuous.
    """
    return jnp.concatenate(
        [
            jnp.atleast_1d(data.qpos[2]),  # torso height
            quat_to_sixd(data.qpos[3:7]),  # torso orientation
            data.qvel[:6],  # torso linear + angular velocity
            data.qpos[7:],  # joint angles
            data.qvel[6:],  # joint velocities
        ]
    )


def joint_limits(mj_model):
    """(low, high) angle per actuator, in radians. The compiler already
    converted jnt_range from the XML's degrees, so this is the one place the
    limits are defined."""
    lo, hi = mj_model.jnt_range[mj_model.actuator_trnid[:, 0]].T
    return jnp.asarray(lo), jnp.asarray(hi)


def act(params, data, limits):
    """Observation -> position target per actuator, squashed into joint range."""
    lo, hi = limits
    u = jnp.tanh(mlp.forward(params, observe(data)))
    return lo + (u + 1) * 0.5 * (hi - lo)
