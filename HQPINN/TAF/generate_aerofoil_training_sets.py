"""
generate_aerofoil_training_sets.py

Generates training datasets for the 2D transonic NACA0012 problem.

Outputs .npy files in TAF/NACA0012/:
  - X_in.npy            (inlet points)            shape (N_in, 2)
  - X_out.npy           (outlet points)           shape (N_out, 2)
  - X_top.npy           (top boundary points)     shape (N_top, 2)
  - X_bot.npy           (bottom boundary points)  shape (N_bot, 2)
  - X_wall.npy          (airfoil surface points)  shape (N_wall, 2)
  - X_wall_normals.npy  (x,y,nx,ny for wall)      shape (N_wall, 4)
  - X_data_int.npy      (internal CFD data points) shape (N_data_int, 2)
  - X_f.npy             (PDE collocation points)  shape (N_pde, 2)

Usage:
  python generate_aerofoil_training_sets.py
Dependencies:
  numpy, (matplotlib optional for quick plot)
"""

import numpy as np
from pathlib import Path

# ----------------------------
# 0) Settings / parameters
# ----------------------------
try:
    # Package execution: python -m HQPINN.HQPINN.TAF.generate_aerofoil_training_sets
    from ..config import (
        TAF_CHORD_X0,
        TAF_CHORD_X1,
        TAF_N_BOUNDARY,
        TAF_N_DATA_INTERNAL,
        TAF_N_DOMAIN_TOTAL,
        TAF_N_WALL,
        TAF_X_MAX,
        TAF_X_MIN,
        TAF_Y_MAX,
        TAF_Y_MIN,
    )
except ImportError:
    try:
        # Direct script execution: ensure HQPINN/HQPINN is importable
        import sys

        pkg_root = Path(__file__).resolve().parents[1]
        if str(pkg_root) not in sys.path:
            sys.path.insert(0, str(pkg_root))

        from config import (
            TAF_CHORD_X0,
            TAF_CHORD_X1,
            TAF_N_BOUNDARY,
            TAF_N_DATA_INTERNAL,
            TAF_N_DOMAIN_TOTAL,
            TAF_N_WALL,
            TAF_X_MAX,
            TAF_X_MIN,
            TAF_Y_MAX,
            TAF_Y_MIN,
        )
    except ImportError:
        # Direct script execution from repo root
        from HQPINN.HQPINN.config import (
            TAF_CHORD_X0,
            TAF_CHORD_X1,
            TAF_N_BOUNDARY,
            TAF_N_DATA_INTERNAL,
            TAF_N_DOMAIN_TOTAL,
            TAF_N_WALL,
            TAF_X_MAX,
            TAF_X_MIN,
            TAF_Y_MAX,
            TAF_Y_MIN,
        )

rng = np.random.default_rng(0)


# ---------------------------------------------------------
# 1) NACA0012 thickness function
# ---------------------------------------------------------
# This function implements the standard NACA 4-digit
# thickness formula for a symmetric airfoil.
#
# For NACA 0012:
#   - "00" → symmetric (no camber)
#   - "12" → 12% maximum thickness
#
# The classical formula is:
#
#   y_t(x) = 5 t [
#       0.2969 √x
#     - 0.1260 x
#     - 0.3516 x²
#     + 0.2843 x³
#     - 0.1015 x⁴
#   ]
#
# with:
#   t = 0.12  (12% thickness)
#   chord c = 1
#
# Since 5 * 0.12 = 0.6, we obtain the factor 0.6 below.
#
# IMPORTANT:
#   x must be in [0, 1], i.e. normalized by the chord.
#   The function returns the HALF-thickness:
#       +y_t → upper surface (extrados)
#       -y_t → lower surface (intrados)
def naca4_thickness(x):
    x = np.asarray(x)  # Ensure vectorized NumPy operations

    return 0.6 * (
        0.2969 * np.sqrt(np.clip(x, 0.0, None))  # √x term (singular slope at LE)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )


# ---------------------------------------------------------
# 2) NACA0012 surface generator (closed polygon)
# ---------------------------------------------------------
# This function builds the FULL airfoil boundary (wall points)
# from the NACA0012 half-thickness distribution.
#
# Geometry convention:
#   - Chord runs from chord_start (LE) to chord_end (TE)
#   - Airfoil is symmetric about y = 0
#   - Thickness function returns half-thickness y_t(x)
#
# The contour is built as:
#   1) Upper surface  (extrados)  : LE → TE
#   2) Lower surface  (intrados)  : TE → LE
#
# This produces a closed loop suitable for:
#   - Wall boundary conditions
#   - Normal computation
#   - Interior masking
def generate_naca0012_surface(
    n_points_along_chord=200, chord_start=None, chord_end=None
):
    if chord_start is None:
        chord_start = TAF_CHORD_X0
    if chord_end is None:
        chord_end = TAF_CHORD_X1
    # Local normalized chord coordinate (0 → 1)
    # x_local is dimensionless (x/c), independent of physical scaling.
    x_local = np.linspace(0.0, 1.0, n_points_along_chord)

    # Compute half-thickness distribution
    yt = naca4_thickness(x_local)

    # Upper surface (extrados)
    # Scale normalized coordinate to physical chord:
    # x = chord_start + x_local * chord_length
    chord_length = chord_end - chord_start

    xu = chord_start + x_local * chord_length
    yu = +yt  # positive offset above camber line (y=0)

    # Lower surface (intrados)
    # Reverse x order so contour closes properly (TE → LE)
    xl = chord_start + x_local[::-1] * chord_length
    yl = -yt[::-1]  # negative offset below camber line

    # Build closed polygon
    xs = np.concatenate([xu, xl])
    ys = np.concatenate([yu, yl])

    return xs, ys


# ---------------------------------------------------------
# 2) Build airfoil wall points and outward normals
# ---------------------------------------------------------
# This block:
#   1) Generates the airfoil boundary points (wall points)
#   2) Computes outward unit normal vectors at each wall point
#   3) Packs everything into (x, y, nx, ny)
#
# These are later used to impose the free-slip condition:
#       u · n = 0
# on the airfoil surface.

# Generate closed airfoil contour:
#   upper surface (LE → TE)
#   lower surface (TE → LE)
# We divide TAF_N_WALL by 2 because half the points go to each surface.
Xw_x, Xw_y = generate_naca0012_surface(
    n_points_along_chord=TAF_N_WALL // 2,
    chord_start=TAF_CHORD_X0,
    chord_end=TAF_CHORD_X1,
)

# Stack coordinates into (x, y) pairs
# Shape: (N_wall_points, 2)
X_wall = np.stack([Xw_x, Xw_y], axis=-1)


# Compute outward unit normals along the airfoil boundary
def compute_normals(xs, ys):

    # Compute numerical tangent vector along the curve
    # This is derivative along the contour parameter (not ∂/∂x or ∂/∂y)
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    tangents = np.stack([dx, dy], axis=-1)

    # Rotate tangent by -90°:
    # If t = (tx, ty), a perpendicular vector is (-ty, tx)
    normals = np.empty_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]

    # Normalize to unit length
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero
    normals /= norms

    # Ensure normals point outward
    # Compute centroid of the polygon
    centroid = np.array([np.mean(xs), np.mean(ys)])

    # Vector from centroid to each boundary point
    vecs = np.stack([xs - centroid[0], ys - centroid[1]], axis=-1)

    # If dot product < 0, normal points inward → flip it
    dotp = np.sum(vecs * normals, axis=1)
    flip_mask = dotp < 0
    normals[flip_mask] *= -1.0

    return normals


# Compute outward normals
Xw_normals = compute_normals(Xw_x, Xw_y)

# Final wall array:
# Each row contains (x, y, nx, ny)
# Used for enforcing: u * nx + v * ny = 0
X_wall_normals = np.concatenate([X_wall, Xw_normals], axis=1)


# ---------------------------------------------------------
# 3) Boundary sampling for the rectangular CFD domain
# ---------------------------------------------------------
# The computational domain is a rectangle:
#   x ∈ [TAF_X_MIN, TAF_X_MAX]
#   y ∈ [TAF_Y_MIN, TAF_Y_MAX]
#
# We discretize each of the four outer boundaries:
#   - Left  boundary  → inlet  (X_in)
#   - Right boundary  → outlet (X_out)
#   - Top boundary    → X_top
#   - Bottom boundary → X_bot
#
# NOTE:
# These are NOT the airfoil surface points.
# The airfoil wall points are generated separately.

# ---- Inlet (left vertical boundary, x = constant = TAF_X_MIN)
# We sample points uniformly along the y-direction.
y_in = np.linspace(TAF_Y_MIN, TAF_Y_MAX, TAF_N_BOUNDARY)

# Create (x, y) pairs:
# x is fixed at TAF_X_MIN
# y varies along the vertical boundary
X_in = np.stack([np.full_like(y_in, TAF_X_MIN), y_in], axis=-1)

# ---- Outlet (right vertical boundary, x = constant = TAF_X_MAX)
# Same y sampling as inlet.
X_out = np.stack([np.full_like(y_in, TAF_X_MAX), y_in], axis=-1)

# ---- Top boundary (horizontal boundary, y = constant = TAF_Y_MAX)
# Here x varies while y is fixed.
x_topbot = np.linspace(TAF_X_MIN, TAF_X_MAX, TAF_N_BOUNDARY)

X_top = np.stack([x_topbot, np.full_like(x_topbot, TAF_Y_MAX)], axis=-1)

# ---- Bottom boundary (horizontal boundary, y = constant = TAF_Y_MIN)
X_bot = np.stack([x_topbot, np.full_like(x_topbot, TAF_Y_MIN)], axis=-1)


# ---------------------------------------------------------
# 4) Point-in-polygon test (Ray Casting Algorithm)
# ---------------------------------------------------------
# This function determines whether each point in xy_points
# lies inside a closed polygon defined by (poly_x, poly_y).
#
# In our case:
#   - The polygon is the airfoil surface.
#   - We use this to remove collocation points that fall
#     inside the solid airfoil (non-physical region).
#
# Method:
#   Ray casting algorithm.
#   For each test point:
#       Cast a horizontal ray to +∞.
#       Count how many times it intersects the polygon edges.
#   If the number of intersections is odd → point is inside.
#   If even → point is outside.
def point_in_polygon(xy_points, poly_x, poly_y):

    # Extract x and y coordinates of test points
    x = xy_points[:, 0]
    y = xy_points[:, 1]

    # Number of vertices in polygon
    n = len(poly_x)

    # Boolean mask: True = inside polygon
    inside = np.zeros(len(x), dtype=bool)

    # Loop over each polygon edge
    for i in range(n):
        j = (i + n - 1) % n  # Previous vertex index (wrap-around)

        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]

        # Check if horizontal ray crosses this edge
        # Condition 1:
        #   The test point's y lies between yi and yj
        # Condition 2:
        #   The intersection x-coordinate is to the right of the point
        intersect = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-20) + xi
        )

        # Toggle inside status each time we detect an intersection
        inside ^= intersect

    return inside


# The airfoil polygon is defined by its boundary coordinates
poly_x = Xw_x
poly_y = Xw_y


# ---------------------------------------------------------
# 5) Domain collocation points (uniform sampling + airfoil filtering)
# ---------------------------------------------------------
# Goal:
#   Build a set of points in the *fluid domain* (rectangle minus airfoil interior).
#
# We create:
#   - X_data_int : internal points supervised by CFD data
#   - X_f        : collocation points used for the PDE residual
#
# Strategy:
#   1) Sample points uniformly in the bounding rectangle.
#   2) Remove points that fall inside the airfoil polygon.
#   3) If we don't have enough points left, sample extra points.
#   4) Shuffle and keep the requested total number.

# Oversample because many points will be rejected (inside the airfoil)
oversample_factor = 2.5
n_try = int(TAF_N_DOMAIN_TOTAL * oversample_factor)

# Uniform random points in the rectangular domain
# points shape: (n_try, 2) with columns (x, y)
points = rng.uniform(
    [TAF_X_MIN, TAF_Y_MIN],
    [TAF_X_MAX, TAF_Y_MAX],
    size=(n_try, 2),
)

# Identify points that lie inside the airfoil polygon (solid region)
inside_mask = point_in_polygon(points, poly_x, poly_y)

# Keep only points outside the airfoil → fluid region
points_outside = points[~inside_mask]

# Safety: if too many points were rejected (airfoil is large or oversampling too small),
# sample more points and filter again until we have enough.
while len(points_outside) < TAF_N_DOMAIN_TOTAL:
    needed = TAF_N_DOMAIN_TOTAL - len(points_outside)

    # Sample extra points (slightly more than needed to reduce chance of shortage)
    extra = rng.uniform(
        [TAF_X_MIN, TAF_Y_MIN],
        [TAF_X_MAX, TAF_Y_MAX],
        size=(int(needed * 1.5) + 100, 2),
    )

    # Filter extra points with the same inside-airfoil test
    extra_inside = point_in_polygon(extra, poly_x, poly_y)
    extra_out = extra[~extra_inside]

    # Append extra valid fluid points to the pool
    points_outside = np.vstack([points_outside, extra_out])

# Keep exactly the requested total number of valid fluid points
points_outside = points_outside[:TAF_N_DOMAIN_TOTAL]

# Shuffle points to avoid any spatial ordering bias
perm = rng.permutation(len(points_outside))
points_outside = points_outside[perm]

if TAF_N_DATA_INTERNAL > TAF_N_DOMAIN_TOTAL:
    raise ValueError(
        "TAF_N_DATA_INTERNAL must be <= TAF_N_DOMAIN_TOTAL "
        f"(got {TAF_N_DATA_INTERNAL} > {TAF_N_DOMAIN_TOTAL})."
    )

# Split internal fluid points:
#   - first chunk for CFD-supervised internal data
#   - remaining chunk for PDE residual points
X_data_int = points_outside[:TAF_N_DATA_INTERNAL]
X_f = points_outside[TAF_N_DATA_INTERNAL:]

# ----------------------------
# 6) Save files and print summary
# ----------------------------
script_path = Path(__file__).resolve()
output_dir = script_path.parent / "NACA0012"
output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir / "X_in.npy", X_in)
np.save(output_dir / "X_out.npy", X_out)
np.save(output_dir / "X_top.npy", X_top)
np.save(output_dir / "X_bot.npy", X_bot)
np.save(output_dir / "X_wall.npy", X_wall)
np.save(output_dir / "X_wall_normals.npy", X_wall_normals)
np.save(output_dir / "X_data_int.npy", X_data_int)
np.save(output_dir / "X_f.npy", X_f)

print(
    "Saved: X_in, X_out, X_top, X_bot, X_wall, X_wall_normals, "
    "X_data_int, X_f"
)
print("Saved to directory:", output_dir)
print(
    "Domain bounds: x_min,x_max =",
    TAF_X_MIN,
    TAF_X_MAX,
    "  y_min,y_max =",
    TAF_Y_MIN,
    TAF_Y_MAX,
)
print("Chord placed on [0,1] (LE at x=0, TE at x=1)")
