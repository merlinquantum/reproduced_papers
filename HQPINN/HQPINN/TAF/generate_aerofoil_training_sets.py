"""
generate_aerofoil_training_sets.py

Generates training datasets for the 2D transonic NACA0012 problem.

Outputs .npy files:
  - X_in.npy            (inlet points)            shape (N_in, 2)
  - X_out.npy           (outlet points)           shape (N_out, 2)
  - X_top.npy           (top boundary points)     shape (N_top, 2)
  - X_bot.npy           (bottom boundary points)  shape (N_bot, 2)
  - X_wall.npy          (airfoil surface points)  shape (N_wall, 2)
  - X_wall_normals.npy  (x,y,nx,ny for wall)      shape (N_wall, 4)
  - X_f.npy             (PDE collocation points)  shape (N_pde, 2)
  - X_data_int.npy      (internal supervised pts) shape (N_data_int, 2)

Usage:
  python generate_aerofoil_training_sets.py
Dependencies:
  numpy, (matplotlib optional for quick plot)
"""

import numpy as np

# ----------------------------
# 0) Settings / parameters
# ----------------------------
from config import *

rng = np.random.default_rng(0)


# ----------------------------
# 1) NACA0012 thickness function & surface generator
# ----------------------------
def naca4_thickness(x):
    """
    Thickness distribution for symmetric NACA 00xx.
    x in [0,1] (local chord coordinate). Returns half-thickness (upper surface offset).
    """
    x = np.asarray(x)
    return 0.6 * (
        0.2969 * np.sqrt(np.clip(x, 0.0, None))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )


def generate_naca0012_surface(
    n_points_along_chord=200, chord_start=TAF_CHORD_X0, chord_end=TAF_CHORD_X1
):
    """
    Generate airfoil coordinates.
    Returns arrays xs, ys making a closed loop: upper surface (LE->TE), lower surface (TE->LE).
    """
    x_local = np.linspace(0.0, 1.0, n_points_along_chord)
    yt = naca4_thickness(x_local)
    xu = chord_start + x_local * (chord_end - chord_start)
    yu = +yt
    xl = chord_start + x_local[::-1] * (chord_end - chord_start)
    yl = -yt[::-1]
    xs = np.concatenate([xu, xl])
    ys = np.concatenate([yu, yl])
    return xs, ys


# ----------------------------
# 2) Build wall points and normals
# ----------------------------
Xw_x, Xw_y = generate_naca0012_surface(
    n_points_along_chord=TAF_N_WALL // 2,
    chord_start=TAF_CHORD_X0,
    chord_end=TAF_CHORD_X1,
)
X_wall = np.stack([Xw_x, Xw_y], axis=-1)


def compute_normals(xs, ys):
    """
    Compute approximate outward normals from the polygon points.
    Uses numerical tangent (gradient), rotates tangent by -90deg, normalizes,
    and orients normals outward using centroid heuristic.
    """
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    tangents = np.stack([dx, dy], axis=-1)
    normals = np.empty_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals /= norms
    centroid = np.array([np.mean(xs), np.mean(ys)])
    vecs = np.stack([xs - centroid[0], ys - centroid[1]], axis=-1)
    dotp = np.sum(vecs * normals, axis=1)
    flip_mask = dotp < 0
    normals[flip_mask] *= -1.0
    return normals


Xw_normals = compute_normals(Xw_x, Xw_y)
X_wall_normals = np.concatenate([X_wall, Xw_normals], axis=1)  # x,y,nx,ny

# ----------------------------
# 3) Boundaries sampling
# ----------------------------
y_in = np.linspace(TAF_Y_MIN, TAF_Y_MAX, TAF_N_BOUNDARY)
X_in = np.stack([np.full_like(y_in, TAF_X_MIN), y_in], axis=-1)

y_out = np.linspace(TAF_Y_MIN, TAF_Y_MAX, TAF_N_BOUNDARY)
X_out = np.stack([np.full_like(y_out, TAF_X_MAX), y_out], axis=-1)

x_topbot = np.linspace(TAF_X_MIN, TAF_X_MAX, TAF_N_BOUNDARY)
X_top = np.stack([x_topbot, np.full_like(x_topbot, TAF_Y_MAX)], axis=-1)
X_bot = np.stack([x_topbot, np.full_like(x_topbot, TAF_Y_MIN)], axis=-1)


# ----------------------------
# 4) Simple point-in-polygon to filter interior points
# ----------------------------
def point_in_polygon(xy_points, poly_x, poly_y):
    """
    Ray-casting algorithm. Returns True for points inside polygon.
    xy_points: (N,2)
    poly_x, poly_y: vertex arrays in order.
    """
    x = xy_points[:, 0]
    y = xy_points[:, 1]
    n = len(poly_x)
    inside = np.zeros(len(x), dtype=bool)
    for i in range(n):
        j = (i + n - 1) % n
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        intersect = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-20) + xi
        )
        inside ^= intersect
    return inside


poly_x = Xw_x
poly_y = Xw_y

# ----------------------------
# 5) Domain collocation points (uniform draw + filter)
# ----------------------------
oversample_factor = 2.5
n_try = int(TAF_N_DOMAIN_TOTAL * oversample_factor)
points = rng.uniform([TAF_X_MIN, TAF_Y_MIN], [TAF_X_MAX, TAF_Y_MAX], size=(n_try, 2))
inside_mask = point_in_polygon(points, poly_x, poly_y)
points_outside = points[~inside_mask]

if len(points_outside) < TAF_N_DOMAIN_TOTAL:
    needed = TAF_N_DOMAIN_TOTAL - len(points_outside)
    extra = rng.uniform(
        [TAF_X_MIN, TAF_Y_MIN],
        [TAF_X_MAX, TAF_Y_MAX],
        size=(int(needed * 1.5) + 100, 2),
    )
    extra_inside = point_in_polygon(extra, poly_x, poly_y)
    extra_out = extra[~extra_inside]
    points_outside = np.vstack([points_outside, extra_out])

points_outside = points_outside[:TAF_N_DOMAIN_TOTAL]
perm = rng.permutation(len(points_outside))
points_outside = points_outside[perm]

X_data_int = points_outside[:TAF_N_DATA_INTERNAL]
X_f = points_outside[TAF_N_DATA_INTERNAL:TAF_N_DOMAIN_TOTAL]

# ----------------------------
# 6) Save files and print summary
# ----------------------------
np.save("X_in.npy", X_in)
np.save("X_out.npy", X_out)
np.save("X_top.npy", X_top)
np.save("X_bot.npy", X_bot)
np.save("X_wall.npy", X_wall)
np.save("X_wall_normals.npy", X_wall_normals)
np.save("X_f.npy", X_f)
np.save("X_data_int.npy", X_data_int)

print("Saved: X_in, X_out, X_top, X_bot, X_wall, X_wall_normals, X_f, X_data_int")
print(
    "Domain bounds: x_min,x_max =",
    TAF_X_MIN,
    TAF_X_MAX,
    "  y_min,y_max =",
    TAF_Y_MIN,
    TAF_Y_MAX,
)
print("Chord placed on [0,1] (LE at x=0, TE at x=1)")
