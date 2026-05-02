"""
generation.py
Generates the initial world state.

Altitude : pure numpy tileable value noise, seamlessly toric.

Tiling fix :
  Each octave uses period = int(freq), which guarantees that
  noise(x=0) == noise(x=freq) on both axes — seamless on the torus.
  All frequencies must be integers >= 2 to avoid degenerate cases.

Octave balance :
  Higher-frequency octaves have larger relative amplitudes than before,
  giving more pronounced peaks and valleys at local scale.

Temperature : latitudinal gradient via cos(π·v).
Fertility : per-base-type noise, varying frequency.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from sim.world import World, BASE_BARE, BASE_SAND, BASE_SOIL


# Octaves : (amplitude, frequency)
# All frequencies must be integers >= 2 for seamless toric tiling.
# Higher amplitudes on mid/high freqs → more peaks and valleys.
OCTAVES = [
    (1.000,  2),    # continental scale  (2 repetitions across map)
    (0.600,  4),    # regional features
    (0.400,  8),    # mountain ranges
    (0.250, 17),    # hills and valleys
    (0.125, 34),    # local relief
    (0.063, 70),    # fine texture
]

BLUR_SIGMA = 0.8   # reduced from 1.5 — keeps more sharp detail


# Fertility noise params per base type : (frequency, blur)
FERTILITY_PARAMS = {
    BASE_BARE : (16, 0.5),
    BASE_SAND : ( 6, 1.5),
    BASE_SOIL : ( 2, 3.0),
}


def generate(world: World, seed: int = 42) -> None:
    _generate_altitude(world, seed)
    _generate_base_type(world)
    _generate_fertility(world, seed)
    _distribute_water(world)
    _distribute_temperature(world)
    _update_albedo(world)


# ------------------------------------------------------------------
# Hash
# ------------------------------------------------------------------

def _hash2(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Fast integer hash → float64 in [-1, 1]. All arithmetic in uint32."""
    ix = ix.astype(np.uint32)
    iy = iy.astype(np.uint32)
    s  = np.uint32(seed & 0xFFFFFFFF)
    h  = ix * np.uint32(1619) + iy * np.uint32(31337) + s * np.uint32(1013)
    h  = ((h >> np.uint32(13)) ^ h)
    h  = h * (h * h * np.uint32(15731) + np.uint32(789221)) + np.uint32(1376312589)
    return (h.astype(np.float64) / np.float64(0xFFFFFFFF)) * 2.0 - 1.0


# ------------------------------------------------------------------
# Seamlessly tiling value noise
# ------------------------------------------------------------------

def _value_noise_2d(h: int, w: int, freq: int, seed: int) -> np.ndarray:
    """
    Tileable 2D value noise on a (h, w) grid.

    Tiling guarantee :
      period = freq (integer).
      The lattice wraps at exactly freq units, so the sample at x=0
      and the sample at x=freq (= next tile start) use the same corner
      hashes → perfectly seamless on a torus.

    freq must be an integer >= 2.
    """
    assert isinstance(freq, int) and freq >= 2, "freq must be an integer >= 2 for seamless tiling"
    period = freq

    # Sample coordinates in [0, freq)
    xs = np.linspace(0.0, float(freq), w, endpoint=False, dtype=np.float64)
    ys = np.linspace(0.0, float(freq), h, endpoint=False, dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys)   # (h, w)

    # Integer lattice corners (wrap at period)
    x0 = np.floor(xg).astype(np.int32) % period
    y0 = np.floor(yg).astype(np.int32) % period
    x1 = (x0 + 1) % period
    y1 = (y0 + 1) % period

    # Fractional part + smoothstep
    fx = xg - np.floor(xg)
    fy = yg - np.floor(yg)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)

    # Bilinear interpolation of corner hashes
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

    for i, (amplitude, freq) in enumerate(OCTAVES):
        layer = _value_noise_2d(h, w, freq, seed + i * 7919)
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
    bt[alt >= 0.8] = BASE_BARE
    world.base_type = bt


# ------------------------------------------------------------------
# Fertility
# ------------------------------------------------------------------

def _generate_fertility(world: World, seed: int) -> None:
    h, w   = world.height, world.width
    bt     = world.base_type
    result = np.zeros((h, w), dtype=np.float32)

    fertility_seed = seed + 99991

    for base_val, (freq, blur) in FERTILITY_PARAMS.items():
        freq = max(2, int(freq))   # ensure integer >= 2
        layer = _value_noise_2d(h, w, freq, fertility_seed + base_val * 3571)
        layer = gaussian_filter(layer, sigma=blur)
        mn, mx = layer.min(), layer.max()
        if mx - mn > 1e-6:
            layer = (layer - mn) / (mx - mn) * 2.0 - 1.0
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
    weight  = np.clip(weight, 0.1, None)
    weight /= weight.sum()
    #world.front.ground_water = (weight * total * n_cells).astype(np.float32)
    # Water is distributed as mist instead of ground water, to give more dynamic range for evaporation in early ticks.
    world.front.mist = 8 * (weight * total * n_cells).astype(np.float32)


# ------------------------------------------------------------------
# Temperature
# ------------------------------------------------------------------

def _distribute_temperature(world: World) -> None:
    alt = world.altitude
    cfg = world.config["atmosphere"]

    v          = world.uv[:, :, 1]
    # World is Toroïdal, so v=0 and v=1 are the same line → cos(2πv) is seamless.
    lat_factor = 1 - np.cos(2 * np.pi * v).astype(np.float32)

    temp_base  = 20.0
    temp_range = 15.0
    alt_lapse  = 15.0

    ground_temp = (
        temp_base + temp_range * lat_factor - alt_lapse * alt
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
    albedo[bt == BASE_BARE] = cfg["bare"]["albedo_base"]
    albedo[bt == BASE_SAND] = cfg["sand"]["albedo_base"]
    albedo[bt == BASE_SOIL] = cfg["soil"]["albedo_base"]
    world.front.albedo = albedo