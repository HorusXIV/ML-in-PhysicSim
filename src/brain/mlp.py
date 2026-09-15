"""Minimal MLP in plain JAX.

Parameters are a list of (W, b) tuples, which is already a pytree, so jax.grad,
jax.jit and jax.vmap work on them with no registration and no framework.
"""

import jax
import jax.numpy as jnp


def init(key, sizes):
    """Draws initial parameters for an MLP.

    Weights are Xavier-scaled normals and biases start at zero.

    Args:
        key: PRNG key used to draw the weight matrices.
        sizes: Layer widths, given as [in_dim, *hidden, out_dim].

    Returns:
        A list of (W, b) tuples, one per layer, where W has shape
        (sizes[i], sizes[i + 1]) and b has shape (sizes[i + 1],).
    """
    return [
        (jax.random.normal(k, (m, n)) * jnp.sqrt(1.0 / m), jnp.zeros(n))
        for k, m, n in zip(jax.random.split(key, len(sizes) - 1), sizes[:-1], sizes[1:])
    ]


def forward(params, x):
    """Runs a forward pass through tanh hidden layers and a linear output.

    Args:
        params: Layer parameters as returned by init.
        x: Input of shape (..., in_dim).

    Returns:
        Output of shape (..., out_dim).
    """
    *hidden, (w_out, b_out) = params
    for w, b in hidden:
        x = jnp.tanh(x @ w + b)
    return x @ w_out + b_out
