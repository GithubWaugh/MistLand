"""
world.py
Defines the World class : holds all static and mutable world data,
manages the double buffer, and drives the tick loop.
"""

import numpy as np
from sim.buffers import WorldBuffers
from sim.phases.evaporation import MIST_UNIT

# Vegetation levels
VEG_NONE    = 0
VEG_LICHENS = 1
VEG_GRASS   = 2
VEG_SHRUBS  = 3
VEG_TREES   = 4

# Base types
BASE_BARE   = 0
BASE_SAND   = 1
BASE_SOIL   = 2

# Set to True to enable per-phase water diagnostic
WATER_DEBUG = True


def _w(buf: WorldBuffers) -> float:
    """Total water in a buffer."""
    return (
        float(buf.ground_water.sum())
        + float(buf.mist.sum()) * MIST_UNIT
        + float(buf.vegetation_water.sum())
    )


class World:
    def __init__(self, config: dict):
        self.config = config

        world_cfg   = config["world"]
        self.width  = world_cfg["grid_width"]
        self.height = world_cfg["grid_height"]
        self.tick_count = 0

        shape = (self.height, self.width)

        self.altitude  = np.zeros(shape, dtype=np.float32)
        self.base_type = np.zeros(shape, dtype=np.uint8)

        self.uv = np.zeros((self.height, self.width, 2), dtype=np.float32)
        xs = np.linspace(0.0, 1.0, self.width,  endpoint=False, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, self.height, endpoint=False, dtype=np.float32)
        self.uv[:, :, 0] = xs[np.newaxis, :]
        self.uv[:, :, 1] = ys[:, np.newaxis]

        self.front = WorldBuffers(self.height, self.width)
        self.back  = WorldBuffers(self.height, self.width)

    def swap_buffers(self) -> None:
        self.front, self.back = self.back, self.front

    def sync_back_from_front(self) -> None:
        self.back.copy_from(self.front)

    def tick(self) -> None:
        from sim.phases import (
            temperature, pressure, vegetation, nutriments,
            evaporation, atmosphere, water, rain,
        )

        self.sync_back_from_front()
        w0 = _w(self.front) if WATER_DEBUG else 0.0

        temperature.step(self);  self.swap_buffers(); self.sync_back_from_front()
        pressure.step(self);     self.swap_buffers(); self.sync_back_from_front()

        # --- Vegetation with detailed diagnostic ---
        w_before_veg = _w(self.front) if WATER_DEBUG else 0.0
        vegetation.step(self);   self.swap_buffers(); self.sync_back_from_front()
        if WATER_DEBUG:
            delta_veg = _w(self.front) - w_before_veg
            if abs(delta_veg) > 0.1:
                print(f"  T{self.tick_count:5d} vegetation : {delta_veg:+.4f}  "
                      f"gw={self.front.ground_water.sum():.2f}  "
                      f"mist={self.front.mist.sum()*MIST_UNIT:.2f}  "
                      f"vegw={self.front.vegetation_water.sum():.2f}")

        nutriments.step(self);   self.swap_buffers(); self.sync_back_from_front()
        evaporation.step(self);  self.swap_buffers(); self.sync_back_from_front()
        atmosphere.step(self);   self.swap_buffers(); self.sync_back_from_front()
        water.step(self);        self.swap_buffers(); self.sync_back_from_front()
        rain.step(self);         self.swap_buffers()

        self.tick_count += 1

        # NaN guard
        for name, arr in [
            ("pressure",    self.front.pressure),
            ("atmo_temp",   self.front.atmo_temp),
            ("ground_temp", self.front.ground_temp),
            ("mist",        self.front.mist),
        ]:
            if np.any(np.isnan(arr)):
                print(f"Tick {self.tick_count} : NaN in {name} !")
                break

    def total_water(self) -> float:
        return (
            float(self.front.ground_water.sum())
            + float(self.front.mist.sum()) * MIST_UNIT
            + float(self.front.vegetation_water.sum())
        )

    def total_energy(self) -> float:
        return (
            float(self.front.ground_temp.sum())
            + float(self.front.atmo_temp.sum())
        )

    def report(self) -> str:
        return (
            f"Tick {self.tick_count:6d} | "
            f"Water {self.total_water():.2f} | "
            f"Energy {self.total_energy():.2f}"
        )