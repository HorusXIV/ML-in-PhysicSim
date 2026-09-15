"""Analytic policy gradients (APG): backprop straight through the simulator.

No RL. No value function, no replay buffer, no advantage estimate. The loss is
a function of the simulated trajectory, and jax.grad differentiates it through
every physics step, because mjx.step is an ordinary pure JAX function.

Three things make that work at all. hexapod.xml pins iterations="1", since
above that MJX wraps its constraint solve in a lax.while_loop, which has no
reverse-mode rule and makes jax.grad fail. Truncated BPTT keeps each update to
HORIZON control steps and carries the final state into the next update behind a
stop_gradient, because backpropagating a whole episode does not work: contact
makes the loss surface stiff and the gradient blows up over long windows. And
the physics scan is rematerialized, recomputed on the backward pass rather than
stored, because reverse mode otherwise keeps every intermediate of every
physics step and that is the binding constraint on a small GPU.

On this hexapod the APG gradient is measurably not a usable training signal.
Against central finite differences along a random parameter direction:

    horizon  physics steps   analytic    finite-diff   rel error
          1             10   4.85e-04       3.02e-02       6119%
          4             40  -2.14e-03      -1.59e-01       7310%
         16            160   5.86e-01      -9.90e-01        269%

Even a one-step horizon is off by 62x, and float64 does not fix it: it shifts
the analytic value by three orders of magnitude, so float32 was mostly
roundoff, and the finite differences then disagree with each other by four
orders and change sign with step size. The loss surface is genuinely
non-smooth, as a foot landing one timestep earlier is a discontinuity and
contact-mode switches swamp the local slope. Empirically walk_loss diverges and
track_gait_loss sits flat, never reaching even the zero-policy baseline.

clone_gait is what actually trains this robot. It uses the same network,
adapter and optimizer, but fits the reference gait supervised and touches no
simulator gradient. The APG path is kept because the measurement is worth
having, and because a smoother model, with no contact or with soft contact, is
exactly where APG does pay off.

Attributes:
    CONTROL_REPEAT: Physics steps held at each control step.
"""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
from mujoco import mjx
from tqdm import tqdm

from brain import adapter, gait, mlp
from brain.rotation import quat_to_matrix

CONTROL_REPEAT = 10


def standing_state(mj_model, settle=500):
    """Settles the model onto the floor.

    Every episode starts from the settled pose rather than from the default
    airborne one, so the first steps carry contact signal instead of free fall.

    Args:
        mj_model: The compiled MuJoCo model.
        settle: Physics steps to run before the state is returned.

    Returns:
        An MjData resting on the floor.
    """
    d = mujoco.MjData(mj_model)
    for _ in range(settle):
        mujoco.mj_step(mj_model, d)
    return d


def make_batch(dx, key, n_envs, jitter=0.02):
    """Replicates one state into a batch with jittered joint angles.

    The jitter is what stops every environment from producing an identical
    gradient.

    Args:
        dx: The mjx.Data to copy.
        key: PRNG key used to draw the jitter.
        n_envs: Number of copies to produce.
        jitter: Standard deviation of the joint-angle noise, in radians.

    Returns:
        An mjx.Data whose leaves carry a leading axis of length n_envs.
    """

    def one(k):
        noise = jitter * jax.random.normal(k, (dx.qpos.shape[0] - 7,))
        return dx.replace(qpos=dx.qpos.at[7:].add(noise))

    return jax.vmap(one)(jax.random.split(key, n_envs))


def rollout(params, state, model, limits, horizon):
    """Scans the policy forward for a fixed number of control steps.

    Each control step holds one action for CONTROL_REPEAT physics steps. The
    physics function is checkpointed so it is recomputed on the backward pass
    instead of stored.

    Args:
        params: MLP parameters as returned by mlp.init.
        state: The mjx.Data to start from.
        model: The mjx.Model to step.
        limits: The (low, high) pair returned by adapter.joint_limits.
        horizon: Number of control steps to simulate.

    Returns:
        A (final_state, qpos) pair, where qpos has shape (horizon, nq) and
        holds the configuration after each control step.
    """
    physics = jax.checkpoint(mjx.step)

    def control_step(s, _):
        ctrl = adapter.act(params, s, limits)
        s, _ = jax.lax.scan(
            lambda s, _: (physics(model, s.replace(ctrl=ctrl)), None),
            s,
            length=CONTROL_REPEAT,
        )
        return s, s.qpos

    return jax.lax.scan(control_step, state, length=horizon)


def _upright(qpos):
    """Measures how closely the torso z-axis points at the sky.

    Args:
        qpos: Configuration of shape (..., nq).

    Returns:
        A value of shape (...) that is 1 when the torso points straight up and
        0 when it is horizontal.
    """
    return quat_to_matrix(qpos[..., 3:7])[..., 2, 2]


def walk_loss(params, batch, model, limits, horizon, height):
    """Scores a from-scratch objective: go forward, stay upright, hold height.

    Args:
        params: MLP parameters as returned by mlp.init.
        batch: Batched mjx.Data to roll out, as returned by make_batch.
        model: The mjx.Model to step.
        limits: The (low, high) pair returned by adapter.joint_limits.
        horizon: Number of control steps to simulate.
        height: Target torso height, in metres.

    Returns:
        A (loss, final_states) pair, where loss is the mean over the batch and
        final_states is the batched mjx.Data at the end of the rollout.
    """

    def one(s):
        final, qs = rollout(params, s, model, limits, horizon)
        forward = final.qpos[0] - s.qpos[0]
        upright = jnp.mean(1.0 - _upright(qs))
        drift = jnp.mean(jnp.square(qs[:, 2] - height))
        return -forward + 2.0 * upright + 5.0 * drift, final

    losses, finals = jax.vmap(one)(batch)
    return jnp.mean(losses), finals


def track_gait_loss(params, batch, model, limits, horizon, height):
    """Scores the policy against the reference gait.

    This is the objective that actually converges, because it gives a dense
    per-joint target instead of asking contact gradients to discover a gait on
    their own.

    Args:
        params: MLP parameters as returned by mlp.init.
        batch: Batched mjx.Data to roll out, as returned by make_batch.
        model: The mjx.Model to step.
        limits: The (low, high) pair returned by adapter.joint_limits.
        horizon: Number of control steps to simulate.
        height: Unused, accepted so the signature matches walk_loss.

    Returns:
        A (loss, final_states) pair, where loss is the mean over the batch and
        final_states is the batched mjx.Data at the end of the rollout.
    """
    dt = CONTROL_REPEAT * 0.002

    def one(s):
        final, qs = rollout(params, s, model, limits, horizon)
        t = s.time + dt * jnp.arange(1, horizon + 1)
        ref = jax.vmap(gait.tripod)(t)
        track = jnp.mean(jnp.square(qs[:, 7:] - ref))
        upright = jnp.mean(1.0 - _upright(qs))
        return track + 0.5 * upright, final

    losses, finals = jax.vmap(one)(batch)
    return jnp.mean(losses), finals


def train(
    mj_model,
    params,
    key,
    loss_fn=track_gait_loss,
    epochs=200,
    horizon=16,
    n_envs=32,
    episode=8,
    lr=3e-3,
    adam_lr=3e-4,
    height=0.215,
):
    """Trains the policy by truncated BPTT through the simulator.

    Every episode updates the environments are reset, so a policy cannot win by
    exploiting one lucky starting state. The optimizer is Muon, which
    orthogonalizes the update for 2D weights and falls back to Adam for 1D
    biases, which is why it takes a second learning rate.

    See the module docstring: on this model the gradient does not carry usable
    signal, and clone_gait is what trains the robot in practice.

    Args:
        mj_model: The compiled MuJoCo model.
        params: Initial MLP parameters as returned by mlp.init.
        key: PRNG key used to draw the environment jitter.
        loss_fn: Objective to differentiate, either track_gait_loss or
            walk_loss.
        epochs: Number of gradient updates.
        horizon: Control steps differentiated per update.
        n_envs: Environments simulated in parallel.
        episode: Updates between environment resets.
        lr: Muon learning rate, used for the 2D weights.
        adam_lr: Adam learning rate, used for the 1D biases.
        height: Target torso height in metres, passed through to loss_fn.

    Returns:
        A (params, history) pair, where history holds the loss at each epoch.
    """
    model = mjx.put_model(mj_model)
    limits = adapter.joint_limits(mj_model)
    start = mjx.put_data(mj_model, standing_state(mj_model))
    optimizer = optax.contrib.muon(learning_rate=lr, adam_learning_rate=adam_lr)
    opt = optimizer.init(params)

    @jax.jit
    def step(params, opt, batch):
        (loss, finals), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, model, limits, horizon, height
        )
        updates, opt = optimizer.update(grads, opt, params)
        params = optax.apply_updates(params, updates)
        return params, opt, jax.lax.stop_gradient(finals), loss

    batch = make_batch(start, key, n_envs)
    history = []
    for epoch in tqdm(range(epochs)):
        params, opt, batch, loss = step(params, opt, batch)
        history.append(float(loss))
        if (epoch + 1) % episode == 0:
            batch = make_batch(start, jax.random.fold_in(key, epoch), n_envs)
    return params, history


def clone_gait(mj_model, params, key, epochs=2000, samples=600, lr=3e-3, adam_lr=3e-4):
    """Fits the policy to the reference gait by supervised regression.

    This deliberately does not touch the simulator's gradient. It is here
    because the APG gradient measurably does not carry usable signal on this
    model: against finite differences it is wrong by 60x at a one-step horizon
    and by 270% at sixteen, in both float32 and float64. See the module
    docstring.

    Rolling the reference out open-loop gives (observation, target) pairs, and
    fitting them is an ordinary well-conditioned regression. The observation
    already contains the current gait target, so the function being learned is
    close to a projection, and it converges in seconds.

    Args:
        mj_model: The compiled MuJoCo model.
        params: Initial MLP parameters as returned by mlp.init.
        key: Unused, accepted so the signature matches train.
        epochs: Number of gradient updates.
        samples: Number of (observation, target) pairs to collect.
        lr: Muon learning rate, used for the 2D weights.
        adam_lr: Adam learning rate, used for the 1D biases.

    Returns:
        A (params, history) pair, where history holds the loss at each epoch.
    """
    limits = adapter.joint_limits(mj_model)
    d = standing_state(mj_model)
    obs, targets = [], []
    for i in range(samples * CONTROL_REPEAT):
        d.ctrl[:] = np.asarray(gait.tripod(d.time))
        mujoco.mj_step(mj_model, d)
        if i % CONTROL_REPEAT == 0:
            obs.append(np.asarray(adapter.observe(d)))
            targets.append(np.asarray(gait.tripod(d.time)))
    x, y = jnp.asarray(np.stack(obs)), jnp.asarray(np.stack(targets))

    optimizer = optax.contrib.muon(learning_rate=lr, adam_learning_rate=adam_lr)
    opt = optimizer.init(params)

    def loss_fn(params):
        return jnp.mean(jnp.square(adapter.to_ctrl(mlp.forward(params, x), limits) - y))

    @jax.jit
    def step(params, opt):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt = optimizer.update(grads, opt, params)
        return optax.apply_updates(params, updates), opt, loss

    history = []
    for _ in range(epochs):
        params, opt, loss = step(params, opt)
        history.append(float(loss))
    return params, history
