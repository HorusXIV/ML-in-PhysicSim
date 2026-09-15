"""The reference gait.

Lives in its own module because both sides need it: adapter feeds the current
target into the observation, and train scores the policy against it. Putting it
in either one would make them import each other.

Attributes:
    PERIOD: Seconds per full gait cycle.
    AMPLITUDE: Base joint deflection of the middle legs, in radians.
"""

import jax.numpy as jnp

PERIOD = 0.4
AMPLITUDE = 0.25

# Phase array and side multipliers (same logic as before)
_GROUP = jnp.array([0.0, jnp.pi, 0.0, jnp.pi, 0.0, jnp.pi])
_SIDE = jnp.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])

# Keep the Cartesian scaling so front/back legs match the middle stride length
_HIP_SCALES = jnp.array([2.0, 1.0, 2.0, 2.0, 1.0, 2.0])

def tripod(t, amplitude=AMPLITUDE, period=PERIOD):
    """
    Returns an 18-element array: [coxa, femur, tibia] per leg.
    """
    p = 2 * jnp.pi * t / period + _GROUP
    
    # Coxa sweeps forward/backward
    coxa = _SIDE * _HIP_SCALES * amplitude * jnp.cos(p)
    
    # Femur lifts up during the swing half
    femur = -(amplitude * 1.5) * jnp.maximum(0.0, jnp.sin(p))
    
    # Tibia curls inward slightly during the swing to aid ground clearance
    tibia = (amplitude * 0.8) * jnp.maximum(0.0, jnp.sin(p))
    
    # Stack along the last axis and flatten to [18]
    return jnp.stack([coxa, femur, tibia], -1).reshape(-1)