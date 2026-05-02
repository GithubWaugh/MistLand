"""
io.py
Save and load world state.

Save format :
  <save_dir>/meta.json   — tick count, grid size, config
  <save_dir>/state.npz   — all WorldBuffers arrays (compressed)
"""

import json
import numpy as np
from pathlib import Path


def save(world, save_dir: str) -> None:
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    meta = {
        "tick_count" : world.tick_count,
        "grid_width" : world.width,
        "grid_height": world.height,
        "config"     : world.config,
    }
    with open(path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    np.savez_compressed(
        str(path / "state.npz"),
        altitude         = world.altitude,
        base_type        = world.base_type,
        fertility        = world.fertility,
        ground_water     = world.front.ground_water,
        nutriments       = world.front.nutriments,
        ground_temp      = world.front.ground_temp,
        albedo           = world.front.albedo,
        vegetation_water = world.front.vegetation_water,
        pressure         = world.front.pressure,
        mist             = world.front.mist,
        atmo_temp        = world.front.atmo_temp,
        wind_x           = world.front.wind_x,
        wind_y           = world.front.wind_y,
        vegetation       = world.front.vegetation,
        veg_tick_counter = world.front.veg_tick_counter,
    )


def load(save_dir: str):
    path = Path(save_dir)

    with open(path / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    from sim.world import World
    world = World(meta["config"])
    world.tick_count = meta["tick_count"]

    data = np.load(str(path / "state.npz"))

    world.altitude  = data["altitude"]
    world.base_type = data["base_type"]
    world.fertility = data["fertility"]

    world.front.ground_water     = data["ground_water"]
    world.front.nutriments       = data["nutriments"]
    world.front.ground_temp      = data["ground_temp"]
    world.front.albedo           = data["albedo"]
    world.front.vegetation_water = data["vegetation_water"]
    world.front.pressure         = data["pressure"]
    world.front.mist             = data["mist"]
    world.front.atmo_temp        = data["atmo_temp"]
    world.front.wind_x           = data["wind_x"]
    world.front.wind_y           = data["wind_y"]
    world.front.vegetation       = data["vegetation"]
    world.front.veg_tick_counter = data["veg_tick_counter"]

    world.sync_back_from_front()
    return world
