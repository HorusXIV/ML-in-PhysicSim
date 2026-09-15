"""6D rotation representation (Zhou et al. 2019) and MuJoCo quaternions.

The 6D form is the first two columns of the rotation matrix. Unlike a
quaternion it is a continuous function of rotation, so a network can regress it
without fighting the double cover, where q and -q denote the same rotation.

MuJoCo quaternions are scalar-first, ordered (w, x, y, z). Every function here
follows a trailing-axis convention and broadcasts over any leading batch
dimensions.
"""

import jax.numpy as jnp


def quat_to_matrix(q):
    """Converts a quaternion to a rotation matrix.

    Args:
        q: Unit scalar-first quaternion of shape (..., 4).

    Returns:
        Rotation matrix of shape (..., 3, 3).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # fmt: off
    return jnp.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)
    # fmt: on


def matrix_to_quat(m):
    """Converts a rotation matrix to a quaternion.

    Uses Shepperd's method: all four branches are built and the one with the
    largest leading term is kept, that term sitting on the diagonal of the
    stacked branches. The naive sqrt(1 + trace) formula loses all precision at
    180-degree rotations, which a policy will hit eventually.

    Args:
        m: Rotation matrix of shape (..., 3, 3).

    Returns:
        Unit scalar-first quaternion of shape (..., 4).
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
    best = jnp.argmax(jnp.diagonal(branches, axis1=-2, axis2=-1), -1)
    q = jnp.take_along_axis(branches, best[..., None, None], -2).squeeze(-2)
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


def sixd_to_matrix(d6):
    """Converts a 6D rotation representation to a rotation matrix.

    The two input vectors are Gram-Schmidt orthonormalized into a frame.

    Args:
        d6: 6D rotation representation of shape (..., 6).

    Returns:
        Rotation matrix of shape (..., 3, 3).
    """
    a, b = d6[..., :3], d6[..., 3:]
    c0 = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (c0 * b).sum(-1, keepdims=True) * c0
    c1 = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
    return jnp.stack([c0, c1, jnp.cross(c0, c1)], -1)


def matrix_to_sixd(m):
    """Converts a rotation matrix to a 6D rotation representation.

    Args:
        m: Rotation matrix of shape (..., 3, 3).

    Returns:
        The first two columns of the matrix, flattened to shape (..., 6).
    """
    return m[..., :, :2].swapaxes(-1, -2).reshape(*m.shape[:-2], 6)


def quat_to_sixd(q):
    """Converts a quaternion to a 6D rotation representation.

    Args:
        q: Unit scalar-first quaternion of shape (..., 4).

    Returns:
        6D rotation representation of shape (..., 6).
    """
    return matrix_to_sixd(quat_to_matrix(q))


def sixd_to_quat(d6):
    """Converts a 6D rotation representation to a quaternion.

    Args:
        d6: 6D rotation representation of shape (..., 6).

    Returns:
        Unit scalar-first quaternion of shape (..., 4).
    """
    return matrix_to_quat(sixd_to_matrix(d6))
