"""
generation.py
Generates the initial world state :
  - altitude via 2D simplex noise + gaussian blur
  - base_type derived from altitude
  - initial water distribution
  - initial temperature distribution
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from opensimplex import OpenSimplex

from sim.world import World, BASE_BARE, BASE_SAND, BASE_SOIL


def generate(world: World, seed: int = 42) -> None:
    """
    Populate world.altitude, world.base_type, and world.front
    with an initial plausible state. Modifies world in-place.

    Args:
        world : the World instance to populate
        seed  : random seed for reproducible generation
    """
    _generate_altitude(world, seed)
    _generate_base_type(world)
    _distribute_water(world)
    _distribute_temperature(world)
    _update_albedo(world)


# ------------------------------------------------------------------
# Altitude
# ------------------------------------------------------------------

def _generate_altitude(world: World, seed: int) -> None:
    """
    Fill world.altitude with layered simplex noise + gaussian blur.
    Result is normalized to [0, 1].
    """
    gen     = OpenSimplex(seed)
    h, w    = world.height, world.width
    alt     = np.zeros((h, w), dtype=np.float32)

    # Layer several octaves of noise for more natural terrain
    octaves = [
        (1.0,  1.0 / max(h, w)),   # large features
        (0.5,  2.0 / max(h, w)),   # medium features
        (0.25, 4.0 / max(h, w)),   # small details
    ]

    for amplitude, frequency in octaves:
        for y in range(h):
            for x in range(w):
                # Wrap coordinates on a torus :
                # map (x, y) to a point on the surface of a 4D torus
                # so that simplex noise wraps seamlessly on both axes.
                u = x / w
                v = y / h
                nx = np.cos(2 * np.pi * u)
                ny = np.sin(2 * np.pi * u)
                nz = np.cos(2 * np.pi * v)
                nw = np.sin(2 * np.pi * v)
                alt[y, x] += amplitude * gen.noise4(nx, ny, nz, nw)

    # Gaussian blur to smooth ridges
    alt = gaussian_filter(alt, sigma=2.0)

    # Normalize to [0, 1]
    alt -= alt.min()
    alt /= alt.max()

    world.altitude = alt.astype(np.float32)


# ------------------------------------------------------------------
# Base type
# ------------------------------------------------------------------

def _generate_base_type(world: World) -> None:
    """
    Assign base type from altitude thresholds.
    Low altitudes → bare (future lake beds)
    Mid altitudes → sand
    High altitudes → soil (richer ground, more complex terrain)
    Thresholds are simple starting points, to be tuned in config later.
    """
    alt = world.altitude
    bt  = np.zeros_like(alt, dtype=np.uint8)

    bt[alt >= 0.3]  = BASE_SAND
    bt[alt >= 0.6]  = BASE_SOIL

    world.base_type = bt


# ------------------------------------------------------------------
# Water
# ------------------------------------------------------------------

def _distribute_water(world: World) -> None:
    """
    Distribute initial water across cells.
    Total water = total_water (from config) spread across all cells,
    weighted inversely by altitude (low areas get more water).
    """
    total   = world.config["world"]["total_water"]
    h, w    = world.height, world.width
    n_cells = h * w

    # Weight : more water in low areas
    weight  = 1.0 - world.altitude           # [0..1], high altitude → low weight
    weight  = np.clip(weight, 0.01, None)    # avoid zeros
    weight /= weight.sum()                   # normalize to sum = 1

    # Distribute total water proportionally
    gw = (weight * total * n_cells).astype(np.float32)
    gw = np.clip(gw, 0.0, 1.0)

    world.front.ground_water = gw


# ------------------------------------------------------------------
# Temperature
# ------------------------------------------------------------------

def _distribute_temperature(world: World) -> None:
    """
    Set initial ground and atmospheric temperatures.
    Simple gradient : warmer at low altitude, cooler at high altitude.
    Values in Celsius.
    """
    alt         = world.altitude
    temp_min    = 0.0    # °C at highest point
    temp_max    = 30.0   # °C at lowest point

    ground_temp = (temp_max - (temp_max - temp_min) * alt).astype(np.float32)
    atmo_temp   = (ground_temp - 5.0).astype(np.float32)  # atmosphere slightly cooler

    world.front.ground_temp = ground_temp
    world.front.atmo_temp   = atmo_temp

    # Initialize pressure from temperature and altitude (same formula as pressure.py)
    cfg     = world.config["atmosphere"]
    P_base  = cfg["P_base"]
    k_temp  = cfg["k_temp"]
    k_alt   = cfg["k_alt"]

    world.front.pressure = (
        P_base
        - k_temp * atmo_temp
        + k_alt  * alt
    ).astype(np.float32)


# ------------------------------------------------------------------
# Albedo
# ------------------------------------------------------------------

def _update_albedo(world: World) -> None:
    """
    Set initial albedo from base type.
    Vegetation albedo reduction is 0 at generation (no vegetation yet).
    """
    cfg     = world.config["base_types"]
    bt      = world.base_type
    albedo  = np.zeros((world.height, world.width), dtype=np.float32)

    albedo[bt == 0] = cfg["bare"]["albedo_base"]
    albedo[bt == 1] = cfg["sand"]["albedo_base"]
    albedo[bt == 2] = cfg["soil"]["albedo_base"]

    world.front.albedo = albedo
