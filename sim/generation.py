"""
generation.py
Generates the initial world state.

Altitude : pure numpy tileable value noise, 5 octaves.
Fertility : static noise field [-1..1], frequency varies by base type :
  - bare → high frequency (strong local contrast, rocky patchwork)
  - sand → medium frequency (larger patches)
  - soil → low frequency (broad fertile / hostile zones)
Temperature : latitudinal gradient via cos(π·v).
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from sim.world import World, BASE_BARE, BASE_SAND, BASE_SOIL


OCTAVES = [
    (1.000, 1.0),
    (0.500, 2.0),
    (0.250, 4.0),
    (0.125, 8.0),
    (0.063, 16.0),
]

BLUR_SIGMA = 1.5

# Fertility noise parameters per base type
# (frequency_multiplier, blur_sigma)
FERTILITY_PARAMS = {
    BASE_BARE : (16.0, 0.5),   # high freq, sharp contrast
    BASE_SAND : (6.0,  1.5),   # medium freq, moderate patches
    BASE_SOIL : (2.0,  3.0),   # low freq, broad zones
}


def generate(world: World, seed: int = 42) -> None:
    _generate_altitude(world, seed)
    _generate_base_type(world)
    _generate_fertility(world, seed)
    _distribute_water(world)
    _distribute_temperature(world)
    _update_albedo(world)


# ------------------------------------------------------------------
# Pure numpy tileable value noise
# ------------------------------------------------------------------

def _hash2(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    ix = ix.astype(np.uint32)
    iy = iy.astype(np.uint32)
    s  = np.uint32(seed & 0xFFFFFFFF)
    h  = ix * np.uint32(1619) + iy * np.uint32(31337) + s * np.uint32(1013)
    h  = ((h >> np.uint32(13)) ^ h)
    h  = h * (h * h * np.uint32(15731) + np.uint32(789221)) + np.uint32(1376312589)
    return (h.astype(np.float64) / np.float64(0xFFFFFFFF)) * 2.0 - 1.0


def _value_noise_2d(h: int, w: int, freq: float, seed: int) -> np.ndarray:
    xs = np.linspace(0.0, freq, w, endpoint=False, dtype=np.float64)
    ys = np.linspace(0.0, freq, h, endpoint=False, dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys)

    period = int(np.ceil(freq)) + 1

    x0 = np.floor(xg).astype(np.int32) % period
    y0 = np.floor(yg).astype(np.int32) % period
    x1 = (x0 + 1) % period
    y1 = (y0 + 1) % period

    fx = xg - np.floor(xg)
    fy = yg - np.floor(yg)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)

    v00 = _hash2(x0, y0, seed)
    v10 = _hash2(x1, y0, seed)
    v01 = _hash2(x0, y1, seed)
    v11 = _hash2(x1, y1, seed)

    return (v00 * (1 - ux) * (1 - uy)
          + v10 *      ux  * (1 - uy)
          + v01 * (1 - ux) *      uy
          + v11 *      ux  *      uy)


# ------------------------------------------------------------------
# Altitude
# ------------------------------------------------------------------

def _generate_altitude(world: World, seed: int) -> None:
    h, w = world.height, world.width
    alt  = np.zeros((h, w), dtype=np.float64)
    for i, (amplitude, freq_mult) in enumerate(OCTAVES):
        layer = _value_noise_2d(h, w, freq_mult, seed + i * 7919)
        alt  += amplitude * layer
    alt = gaussian_filter(alt, sigma=BLUR_SIGMA)
    alt -= alt.min()
    alt /= alt.max()
    world.altitude = alt.astype(np.float32)


# ------------------------------------------------------------------
# Base type
# ------------------------------------------------------------------

def _generate_base_type(world: World) -> None:
    alt = world.altitude
    bt  = np.zeros_like(alt, dtype=np.uint8)
    bt[alt >= 0.3] = BASE_SAND
    bt[alt >= 0.6] = BASE_SOIL
    world.base_type = bt


# ------------------------------------------------------------------
# Fertility
# ------------------------------------------------------------------

def _generate_fertility(world: World, seed: int) -> None:
    """
    Generate a static fertility field [-1..1] per cell.
    Each base type uses a different noise frequency and blur :
      - bare : high frequency → sharp rocky patchwork
      - sand : medium frequency → dune-like patches
      - soil : low frequency → broad fertile / barren zones
    The final field is a weighted blend — each cell uses only
    the noise layer matching its base type.
    """
    h, w   = world.height, world.width
    bt     = world.base_type
    result = np.zeros((h, w), dtype=np.float32)

    # Use a seed offset far from altitude seeds to avoid correlation
    fertility_seed = seed + 99991

    for base_val, (freq, blur) in FERTILITY_PARAMS.items():
        # Generate noise layer for this base type
        layer = _value_noise_2d(h, w, freq, fertility_seed + base_val * 3571)
        layer = gaussian_filter(layer, sigma=blur)

        # Normalise to [-1, 1]
        mn, mx = layer.min(), layer.max()
        if mx - mn > 1e-6:
            layer = (layer - mn) / (mx - mn) * 2.0 - 1.0

        # Apply only to cells of matching base type
        mask = (bt == base_val)
        result[mask] = layer[mask].astype(np.float32)

    world.fertility = result


# ------------------------------------------------------------------
# Water
# ------------------------------------------------------------------

def _distribute_water(world: World) -> None:
    total   = world.config["world"]["total_water"]
    h, w    = world.height, world.width
    n_cells = h * w
    weight  = 1.0 - world.altitude
    weight  = np.clip(weight, 0.01, None)
    weight /= weight.sum()
    world.front.ground_water = (weight * total * n_cells).astype(np.float32)


# ------------------------------------------------------------------
# Temperature
# ------------------------------------------------------------------

def _distribute_temperature(world: World) -> None:
    alt = world.altitude
    cfg = world.config["atmosphere"]

    v          = world.uv[:, :, 1]
    lat_factor = np.cos(np.pi * v).astype(np.float32)

    temp_base  = 20.0
    temp_range = 15.0
    alt_lapse  = 15.0

    ground_temp = (
        temp_base
        + temp_range * lat_factor
        - alt_lapse  * alt
    ).astype(np.float32)

    atmo_temp = (ground_temp - 5.0).astype(np.float32)

    world.front.ground_temp = ground_temp
    world.front.atmo_temp   = atmo_temp

    P_base   = cfg["P_base"]
    k_temp   = cfg["k_temp"]
    k_alt    = cfg["k_alt"]
    temp_ref = cfg["temp_ref"]

    world.front.pressure = (
        P_base
        - k_temp * (atmo_temp - temp_ref)
        + k_alt  * alt
    ).astype(np.float32)


# ------------------------------------------------------------------
# Albedo
# ------------------------------------------------------------------

def _update_albedo(world: World) -> None:
    cfg    = world.config["base_types"]
    bt     = world.base_type
    albedo = np.zeros((world.height, world.width), dtype=np.float32)
    albedo[bt == 0] = cfg["bare"]["albedo_base"]
    albedo[bt == 1] = cfg["sand"]["albedo_base"]
    albedo[bt == 2] = cfg["soil"]["albedo_base"]
    world.front.albedo = albedo