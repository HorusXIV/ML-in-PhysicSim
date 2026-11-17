# Hexapod locomotion with MuJoCo MJX

An experimental hexapod controller built with JAX and MuJoCo MJX. A small MLP
maps observations to joint controls. The project explores supervised cloning
of a reference tripod gait and analytic policy gradients through the simulator.

The gait-cloning experiment walks; training directly through contact dynamics
has not produced a useful locomotion policy in the documented experiments.
This repository includes the implementation, experiments, and checks rather
than a pretrained controller or a claim of general-purpose locomotion.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

The default dependency groups include development tools, Jupyter, and CUDA 12
JAX wheels. For CPU-only development:

```bash
uv sync --no-group cuda --no-group notebook
```

To use the notebook on CPU, keep its dependency group:

```bash
uv sync --no-group cuda
cd notebooks
uv run --no-group cuda jupyter notebook hexapod.ipynb
```

The notebook's model paths are relative to `notebooks/`. Run it with that
working directory. To use CUDA, omit `--no-group cuda` on a compatible system.
Rendering also needs a suitable MuJoCo/OpenGL backend for your machine.

## Layout

```text
models/SELF_Hexapod/    hexapod.xml and scene.xml
src/brain/
  mlp.py                policy network (JAX parameter pytrees)
  rotation.py           quaternion / continuous 6D rotation conversions
  adapter.py            observations and action-to-control mapping
  movement/gait.py      reference tripod gait
  movement/jump.py      experimental jump motion
  train.py              gait cloning and analytic policy gradients
notebooks/hexapod.ipynb experiments, rollouts, and visualizations
tests/test_brain.py    rotation, simulation, gait, and gradient checks
```

The project installs `src/brain` as the `brain` package.

## Training example

From the repository root, after setup:

```python
import jax
import mujoco
from brain import adapter, mlp
from brain.train import clone_gait

model = mujoco.MjModel.from_xml_path("models/SELF_Hexapod/scene.xml")
params = mlp.init(jax.random.key(0), [adapter.obs_dim(model), 64, 64, model.nu])
params, history = clone_gait(model, params, jax.random.key(1))
```

See the notebook for evaluation and visualization. The existing experiment
reports around 17 seconds for cloning and 1.41 m of travel in 5 seconds;
these are recorded measurements, not a cross-hardware benchmark.

## Experimental findings and limitations

`train()` differentiates through `mjx.step` using truncated backpropagation
through time. It uses no value function or replay buffer. The experiments
compare simulator gradients with finite differences and find large
disagreement around contact changes. The forward-velocity objective diverges;
reference-gait tracking has not beaten the zero-policy baseline in the
reported runs. See `brain.train` and the notebook for the measurements.

The optimizer is `optax.contrib.muon`, using Muon for 2D weights and Adam for
1D biases. Training accepts separate learning rates for these parameter groups.

## Checks

CPU-only commands, run from the repository root:

```bash
uv run --no-group cuda --no-group notebook pytest -q
uv run --no-group cuda --no-group notebook ruff check src tests
uv run --no-group cuda --no-group notebook ruff format --check src tests
```

Known issue at preparation time: `tests/test_brain.py` still imports `gait`
from `brain`; its current location is `brain.movement.gait`. That import must
be updated before the tests can collect. The CI sync step selects CPU-only
dependencies, but its later `uv run` commands also need the same group flags
to avoid requesting default CUDA/notebook groups again.

## Implementation notes

- On a small GPU, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing
  JAX, as the notebook does, to leave memory for MuJoCo rendering.
- The model sets solver `iterations="1"` for the differentiated experiment.
  In the tested setup, higher values use a loop without reverse-mode support.
- The reference motion is `brain.movement.gait.tripod`. Joint sign conventions
  were checked by measuring open-loop travel.
- Joint limits are declared in the model and read by `brain.joint_limits`.
  Actuator `ctrlrange` is intentionally not used as a second set of limits.
