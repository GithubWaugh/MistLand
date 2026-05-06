"""
generation.py
Generates the initial world state.
Wind is initialised from the initial pressure gradient so the simulation
starts with a physically plausible wind field rather than zero.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from sim import world
from sim.world import World, BASE_BARE, BASE_SAND, BASE_SOIL


OCTAVES = [
    (1.000,  2),
    (0.600,  4),
    (0.400,  8),
    (0.250, 16),
    (0.150, 32),
    (0.080, 64),
]

BLUR_SIGMA = 0.8

FERTILITY_PARAMS = {
    BASE_BARE : (256, 0.25),
    BASE_SAND : ( 128, 1.5),
    BASE_SOIL : ( 64, 3.0),
}


def generate(world: World, seed: int = 42) -> None:
    _generate_altitude(world, seed)
    _generate_base_type(world, seed)
    _generate_fertility(world, seed)
    _distribute_water(world)
    _distribute_temperature(world)
    _update_albedo(world)
    _init_wind(world)   # must be last — uses pressure computed in _distribute_temperature
    world.initial_energy = world.total_energy()


# ------------------------------------------------------------------
# Hash + noise
# ------------------------------------------------------------------

def _hash2(ix, iy, seed):
    ix = ix.astype(np.uint32);  iy = iy.astype(np.uint32)
    s  = np.uint32(seed & 0xFFFFFFFF)
    h  = ix * np.uint32(1619) + iy * np.uint32(31337) + s * np.uint32(1013)
    h  = ((h >> np.uint32(13)) ^ h)
    h  = h * (h * h * np.uint32(15731) + np.uint32(789221)) + np.uint32(1376312589)
    return (h.astype(np.float64) / np.float64(0xFFFFFFFF)) * 2.0 - 1.0


def _value_noise_2d(h, w, freq, seed):
    assert freq >= 2
    xs = np.linspace(0.0, float(freq), w, endpoint=False, dtype=np.float64)
    ys = np.linspace(0.0, float(freq), h, endpoint=False, dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys)
    period  = freq
    x0 = np.floor(xg).astype(np.int32) % period
    y0 = np.floor(yg).astype(np.int32) % period
    x1 = (x0 + 1) % period;  y1 = (y0 + 1) % period
    fx = xg - np.floor(xg);  fy = yg - np.floor(yg)
    ux = fx*fx*(3-2*fx);      uy = fy*fy*(3-2*fy)
    v00 = _hash2(x0,y0,seed); v10 = _hash2(x1,y0,seed)
    v01 = _hash2(x0,y1,seed); v11 = _hash2(x1,y1,seed)
    return (v00*(1-ux)*(1-uy) + v10*ux*(1-uy) +
            v01*(1-ux)*uy     + v11*ux*uy)


# ------------------------------------------------------------------
# Altitude
# ------------------------------------------------------------------

def _generate_altitude(world, seed):
    h, w = world.height, world.width
    alt  = np.zeros((h, w), dtype=np.float64)
    for i, (amp, freq) in enumerate(OCTAVES):
        alt += amp * _value_noise_2d(h, w, freq, seed + i*7919)
    alt = gaussian_filter(alt, sigma=BLUR_SIGMA)
    alt -= alt.min();  alt /= alt.max()
    world.altitude = alt.astype(np.float32)


# ------------------------------------------------------------------
# Base type
# ------------------------------------------------------------------

def _generate_base_type(world, seed):
    h, w = world.height, world.width
    noise = np.zeros((h, w), dtype=np.float64)
    bseed = seed + 200003
    for i, (amp, freq) in enumerate(OCTAVES):
        noise += amp * _value_noise_2d(h, w, freq, bseed + i*7919)
    noise = gaussian_filter(noise, sigma=BLUR_SIGMA)
    noise -= noise.min();  noise /= noise.max()
    bt = np.zeros((h, w), dtype=np.uint8)
    bt[noise >= 0.3] = BASE_SAND
    bt[noise >= 0.6] = BASE_SOIL
    world.base_type = bt


# ------------------------------------------------------------------
# Fertility
# ------------------------------------------------------------------

def _generate_fertility(world, seed):
    h, w   = world.height, world.width
    bt     = world.base_type
    result = np.zeros((h, w), dtype=np.float32)
    fseed  = seed + 99991
    for base_val, (freq, blur) in FERTILITY_PARAMS.items():
        freq  = max(2, int(freq))
        layer = _value_noise_2d(h, w, freq, fseed + base_val*3571)
        layer = gaussian_filter(layer, sigma=blur)
        mn, mx = layer.min(), layer.max()
        if mx - mn > 1e-6:
            layer = (layer - mn) / (mx - mn) * 2.0 - 1.0
        result[bt == base_val] = layer[bt == base_val].astype(np.float32)
    world.fertility = abs(result)


# ------------------------------------------------------------------
# Water
# ------------------------------------------------------------------

def _distribute_water(world):
    total   = world.config["world"]["total_water"]
    h, w    = world.height, world.width
    weight  = np.clip(1-world.altitude, 0.01, None)
    weight /= weight.sum()
    world.front.ground_water = (weight * total * h * w).astype(np.float32)


# ------------------------------------------------------------------
# Temperature + pressure
# ------------------------------------------------------------------

def _distribute_temperature(world):
    alt = world.altitude
    cfg = world.config["atmosphere"]

    v          = world.uv[:, :, 1]
    lat_factor = np.cos(np.pi * v * 2 + np.pi).astype(np.float32)

    ground_temp = (20.0 + 15.0 * lat_factor - 15.0 * alt).astype(np.float32)
    atmo_temp   = (ground_temp - 5.0).astype(np.float32)

    world.front.ground_temp = ground_temp
    world.front.atmo_temp   = atmo_temp

    P_base   = cfg["P_base"]
    k_temp   = cfg["k_temp"]
    k_alt    = cfg["k_alt"]
    temp_ref = cfg["temp_ref"]

    world.front.pressure = (
        P_base - k_temp * (atmo_temp - temp_ref) + k_alt * alt
    ).astype(np.float32)


# ------------------------------------------------------------------
# Albedo
# ------------------------------------------------------------------

def _update_albedo(world):
    cfg    = world.config["base_types"]
    bt     = world.base_type
    albedo = np.zeros((world.height, world.width), dtype=np.float32)
    albedo[bt == BASE_BARE] = cfg["bare"]["albedo_base"]
    albedo[bt == BASE_SAND] = cfg["sand"]["albedo_base"]
    albedo[bt == BASE_SOIL] = cfg["soil"]["albedo_base"]
    world.front.albedo = albedo


# ------------------------------------------------------------------
# Wind initialisation from pressure gradient
# ------------------------------------------------------------------

def _init_wind(world):
    """
    Bootstrap the wind field from the initial pressure gradient.
    wind = -k_wind * ∇P  (pointing from high to low pressure)
    This avoids a cold-start with zero wind everywhere.
    """
    P    = world.front.pressure
    k    = world.config["atmosphere"].get("k_wind", 0.1)

    dPdx = (np.roll(P, -1, axis=1) - np.roll(P, 1, axis=1)) * 0.5
    dPdy = (np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)) * 0.5

    world.front.wind_x = (-k * dPdx).astype(np.float32)
    world.front.wind_y = (-k * dPdy).astype(np.float32)
