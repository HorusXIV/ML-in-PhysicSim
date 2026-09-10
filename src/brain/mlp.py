"""Minimal MLP in plain JAX.

Params are a list of (W, b) tuples, which is already a pytree, so jax.grad,
jax.jit and jax.vmap work on them with no registration and no framework.
"""

import jax
import jax.numpy as jnp


def init(key, sizes):
    """sizes = [in_dim, *hidden, out_dim]. Xavier-scaled normal, zero biases."""
    return [
        (jax.random.normal(k, (m, n)) * jnp.sqrt(1.0 / m), jnp.zeros(n))
        for k, m, n in zip(jax.random.split(key, len(sizes) - 1), sizes[:-1], sizes[1:])
    ]


def forward(params, x):
    """tanh hidden layers, linear output."""
    *hidden, (w_out, b_out) = params
    for w, b in hidden:
        x = jnp.tanh(x @ w + b)
    return x @ w_out + b_out
