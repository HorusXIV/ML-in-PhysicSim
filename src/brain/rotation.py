"""6D rotation representation (Zhou et al. 2019) <-> MuJoCo quaternions.

The 6D form is the first two columns of the rotation matrix. Unlike quaternions
it is a continuous function of rotation, so a network can regress it without
fighting the double cover (q and -q are the same rotation).

MuJoCo quaternions are scalar-first: (w, x, y, z). All functions take a
trailing-axis convention and broadcast over any leading batch dims.
"""

import jax.numpy as jnp


def quat_to_matrix(q):
    """(..., 4) scalar-first quaternion -> (..., 3, 3) rotation matrix."""
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # fmt: off
    return jnp.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)
    # fmt: on


def matrix_to_quat(m):
    """(..., 3, 3) rotation matrix -> (..., 4) scalar-first quaternion.

    Shepperd's method: build all four branches and keep the one with the
    largest leading term. The naive sqrt(1 + trace) formula loses all
    precision at 180-degree rotations, which a policy will hit eventually.
    """
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    # fmt: off
    branches = jnp.stack([
        jnp.stack([1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01], -1),
        jnp.stack([m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20], -1),
        jnp.stack([m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21], -1),
        jnp.stack([m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22], -1),
    ], -2)
    # fmt: on
    # Branch i's leading term sits at component i, i.e. on the diagonal.
    best = jnp.argmax(jnp.diagonal(branches, axis1=-2, axis2=-1), -1)
    q = jnp.take_along_axis(branches, best[..., None, None], -2).squeeze(-2)
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


def sixd_to_matrix(d6):
    """(..., 6) -> (..., 3, 3). Gram-Schmidt the two vectors into a frame."""
    a, b = d6[..., :3], d6[..., 3:]
    c0 = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (c0 * b).sum(-1, keepdims=True) * c0
    c1 = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
    return jnp.stack([c0, c1, jnp.cross(c0, c1)], -1)


def matrix_to_sixd(m):
    """(..., 3, 3) -> (..., 6). First two columns, flattened."""
    return m[..., :, :2].swapaxes(-1, -2).reshape(*m.shape[:-2], 6)


def quat_to_sixd(q):
    return matrix_to_sixd(quat_to_matrix(q))


def sixd_to_quat(d6):
    return matrix_to_quat(sixd_to_matrix(d6))
