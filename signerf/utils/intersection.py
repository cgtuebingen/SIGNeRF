""" This module contains functions for computing the intersection of rays with different geometries """

import torch

def intersect_with_aabb(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    aabb: torch.Tensor,
):
    """
    Intersects the rays with a custom box and returns the near and far values.

    Args:
        rays_o: origins of rays [H, W, 3]
        rays_d: directions of rays [H, W, 3]
        aabb: This is [min point (x,y,z), max point (x,y,z)]

    Returns:
        nears: near values for each ray [H, W, 3]
        fars: far values for each ray [H, W, 3]
    """

    # Save the original shape
    original_shape = rays_o.shape

    # Flatten the rays
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)


    # avoid divide by zero
    dir_fraction = 1.0 / (rays_d + 1e-6)

    # x
    t1 = (aabb[0, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
    t2 = (aabb[1, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
    # y
    t3 = (aabb[0, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
    t4 = (aabb[1, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
    # z
    t5 = (aabb[0, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
    t6 = (aabb[1, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]

    nears = torch.max(torch.cat([torch.minimum(t1, t2), torch.minimum(t3, t4), torch.minimum(t5, t6)], dim=1), dim=1).values
    fars = torch.min(torch.cat([torch.maximum(t1, t2), torch.maximum(t3, t4), torch.maximum(t5, t6)], dim=1), dim=1).values

    # clamp to near plane
    #near_plane = 0
    #nears = torch.clamp(nears, min=near_plane)
    #fars = torch.maximum(fars, nears + 1e-6)

    # Reshape the nears and fars
    nears = nears.reshape(original_shape[0], original_shape[1], 1)
    fars = fars.reshape(original_shape[0], original_shape[1], 1)

    return nears, fars

def intersect_with_sphere(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    center: torch.Tensor,
    radius: float,
):
    """
    Intersects the rays with a custom sphere and returns the near and far values.

    Args:
        rays_o: origins of rays [H, W, 3]
        rays_d: directions of rays [H, W, 3]
        center: center of the sphere[x,y,z]
        radius: radius of the sphere

    Returns:
        nears: near values for each ray [H, W, 3]
        fars: far values for each ray [H, W, 3]
    """

    # Save the original shape
    original_shape = rays_o.shape

    # Flatten the rays
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    # Compute the coefficients of the quadratic equation
    oc = rays_o - center
    b = torch.sum(oc * rays_d, dim=1)
    c = torch.sum(oc * oc, dim=1) - radius * radius

    # Compute the discriminant
    discriminant = b * b - c

    # Find the roots of the quadratic equation
    mask = discriminant > 0
    roots = torch.zeros_like(discriminant)
    roots[mask] = torch.sqrt(discriminant[mask])
    t1 = -b - roots
    t2 = -b + roots

    # Compute the near and far values
    t_min = torch.min(t1, t2)
    t_max = torch.max(t1, t2)

    # Clamp the near and far values to avoid negative values
    nears = torch.clamp(t_min, min=0)
    fars = torch.clamp(t_max, min=0)

    # Reshape the nears and fars
    nears = nears.reshape(original_shape[0], original_shape[1], 1)
    fars = fars.reshape(original_shape[0], original_shape[1], 1)

    return nears, fars
