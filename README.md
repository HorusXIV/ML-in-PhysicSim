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
src/brain/              the policy: mlp, 6D<->quat rotation, MuJoCo adapter
notebooks/hexapod.ipynb rollouts, gradients, visualisation
```

`src/brain` installs as the `brain` package, so `import brain` works anywhere in
the venv — no `sys.path` juggling in notebooks.

## Checks

```bash
uv run pytest
uv run ruff check src && uv run ruff format --check src
```

## Notes

- `hexapod.xml` sets `iterations="1"`. MJX only differentiates its constraint
  solver in that case; above 1 it uses a `lax.while_loop`, which has no
  reverse-mode rule, and `jax.grad` through `mjx.step` fails. On this scene the
  accurate solver is indistinguishable anyway.
- Actuators carry no `ctrlrange`: `angle="degree"` converts `jnt_range` but
  *not* `ctrlrange`, so limits are declared once on the joints and
  `brain.joint_limits` reads them back in radians.
