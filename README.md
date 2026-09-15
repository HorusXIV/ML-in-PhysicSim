# physics-sim

Differentiable hexapod locomotion with MuJoCo MJX.

## Setup

```bash
uv sync          # includes CUDA wheels + Jupyter by default
```

CPU only (what CI runs):

```bash
uv sync --no-group cuda --no-group notebook
```

## Layout

```
models/SELF_Hexapod/    hexapod.xml (body) + scene.xml (floor, lights, cameras)
src/brain/
  mlp.py                the policy network (plain JAX, params are a pytree)
  rotation.py           6D <-> quaternion, for continuous orientation inputs
  adapter.py            observation extraction and action -> ctrl mapping
  gait.py               the reference tripod gait
  train.py              analytic policy gradients through MJX
tests/                  unit tests
notebooks/hexapod.ipynb rollouts, gradients, training, visualisation
```

`src/brain` installs as the `brain` package, so `import brain` works anywhere in
the venv — no `sys.path` juggling in notebooks.

## Checks

```bash
uv run pytest
uv run ruff check src tests && uv run ruff format --check src tests
```

## Training

```python
from brain.train import clone_gait
params, history = clone_gait(mj_model, params, jax.random.key(0))
```

`clone_gait` fits the policy to the reference gait supervised: ~17 s, and the
result walks 1.41 m in 5 s.

`train()` is the analytic-policy-gradient version — loss differentiated
straight through `mjx.step`, truncated BPTT, no RL. It is built and it runs,
but on this model its gradient is not a usable signal. Against finite
differences it is wrong by 62x at a one-step horizon and 269% at sixteen, in
both float32 and float64, because contact make/break makes the loss surface
non-smooth. `walk_loss` diverges; `track_gait_loss` plateaus above the
zero-policy baseline. The numbers are in the `brain.train` docstring.

Optimizer is `optax.contrib.muon`. Muon orthogonalizes the update for 2D
weights via Newton-Schulz and falls back to Adam for 1D biases, which is why
`train` takes two learning rates.

## Notes

- On a small GPU set `XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing
  jax. JAX otherwise grabs ~75% of VRAM on first use, leaving cuSolver unable
  to create its handle (`gpusolverDnCreate failed`) and the EGL renderer short
  of memory. The notebook's first cell does this.

- `hexapod.xml` sets `iterations="1"`. MJX only differentiates its constraint
  solver in that case; above 1 it uses a `lax.while_loop`, which has no
  reverse-mode rule, and `jax.grad` through `mjx.step` fails. On this scene the
  accurate solver is indistinguishable anyway.
- The default objective tracks a hand-written tripod gait
  (`brain.gait.tripod`) instead of rewarding forward velocity. This is not a
  shortcut: contact gradients through a planted foot are near zero, so the
  from-scratch objective (`walk_loss`) diverges. DeepMind's own APG example
  tracks a reference gait for the same reason.
- The gait's sign conventions were measured, not derived -- a sweep over both
  sign choices scored on distance travelled open-loop. The wrong knee sign jams
  the left knees against their `-60..0` limit and the robot shuffles in place.
- Actuators carry no `ctrlrange`: `angle="degree"` converts `jnt_range` but
  *not* `ctrlrange`, so limits are declared once on the joints and
  `brain.joint_limits` reads them back in radians.
