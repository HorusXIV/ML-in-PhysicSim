"""Glue between the hexapod and the MLP.

Works on either an MjData (numpy) or an mjx.Data (jax), since both expose qpos
and qvel with the same layout and jnp handles either array type. For a
free-jointed body that layout is qpos = [xyz(3), quat(4), joints...] and
qvel = [linvel(3), angvel(3), joints...].
"""

import jax.numpy as jnp

from brain import gait, mlp
from brain.rotation import quat_to_sixd


def obs_dim(mj_model):
    """Reports the length of the vector returned by observe.

    Args:
        mj_model: The compiled MuJoCo model.

    Returns:
        Height (1) plus orientation (6) plus root velocity (6) plus joint
        angles, joint velocities and the current gait target, the last of which
        holds one entry per actuator.
    """
    return 13 + (mj_model.nq - 7) + (mj_model.nv - 6) + mj_model.nu


def observe(data):
    """Collects everything the brain senses into one flat vector.

    The entries run in order: torso height, torso orientation, torso linear and
    angular velocity, joint angles, joint velocities, and the gait target for
    the current simulation time.

    Absolute x and y are deliberately left out, because a gait should not
    depend on where in the world it started, and orientation goes in as 6D
    rather than as the raw quaternion so the input is continuous. The current
    gait target is included because without it the policy is asked to track a
    time-varying reference it cannot see, and can do no better than the cycle
    average, which is exactly what happened when it was left out. DeepMind's
    APG example feeds kin_ref the same way.

    Args:
        data: An MjData or mjx.Data holding the current simulator state.

    Returns:
        Observation of shape (obs_dim(mj_model),).
    """
    return jnp.concatenate(
        [
            data.qpos[2:3],
            quat_to_sixd(data.qpos[3:7]),
            data.qvel[:6],
            data.qpos[7:],
            data.qvel[6:],
            gait.tripod(data.time),
        ]
    )


def joint_limits(mj_model):
    """Reads the angle limits of the actuated joints out of the model.

    The compiler has already converted jnt_range from the degrees written in
    the XML, so this is the one place the limits are defined.

    Args:
        mj_model: The compiled MuJoCo model.

    Returns:
        A (low, high) pair of arrays, each of shape (mj_model.nu,), giving the
        limits per actuator in radians.
    """
    lo, hi = mj_model.jnt_range[mj_model.actuator_trnid[:, 0]].T
    return jnp.asarray(lo), jnp.asarray(hi)


def to_ctrl(raw, limits):
    """Squashes raw network output into a position target within joint range.

    Args:
        raw: Unbounded network output of shape (..., nu).
        limits: The (low, high) pair returned by joint_limits.

    Returns:
        Position targets of shape (..., nu), inside the joint limits.
    """
    lo, hi = limits
    return lo + (jnp.tanh(raw) + 1) * 0.5 * (hi - lo)


def act(params, data, limits):
    """Maps the current simulator state to a position target per actuator.

    Args:
        params: MLP parameters as returned by mlp.init.
        data: An MjData or mjx.Data holding the current simulator state.
        limits: The (low, high) pair returned by joint_limits.

    Returns:
        Position targets of shape (nu,).
    """
    return to_ctrl(mlp.forward(params, observe(data)), limits)
