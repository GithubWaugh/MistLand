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

WATER_DEBUG = False


class World:
    def __init__(self, config: dict):
        self.config = config

        world_cfg   = config["world"]
        self.width  = world_cfg["grid_width"]
        self.height = world_cfg["grid_height"]
        self.tick_count = 0

        shape = (self.height, self.width)

        # --- Static arrays ---
        self.altitude   = np.zeros(shape, dtype=np.float32)
        self.base_type  = np.zeros(shape, dtype=np.uint8)

        # Fertility : static float32 [-1..1]
        # Positive = locally favourable to life
        # Negative = locally hostile
        # Generated in generation.py, varies by base type :
        #   bare → high-frequency noise (strong local contrast)
        #   sand → mid-frequency noise  (medium patches)
        #   soil → low-frequency noise  (large fertile zones)
        self.fertility  = np.zeros(shape, dtype=np.float32)

        # UV coordinates
        self.uv = np.zeros((self.height, self.width, 2), dtype=np.float32)
        xs = np.linspace(0.0, 1.0, self.width,  endpoint=False, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, self.height, endpoint=False, dtype=np.float32)
        self.uv[:, :, 0] = xs[np.newaxis, :]
        self.uv[:, :, 1] = ys[:, np.newaxis]

        # Double buffer
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

        temperature.step(self);  self.swap_buffers(); self.sync_back_from_front()
        pressure.step(self);     self.swap_buffers(); self.sync_back_from_front()
        vegetation.step(self);   self.swap_buffers(); self.sync_back_from_front()
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

    def get_flooded_mask(self) -> np.ndarray:
        base_cfg = self.config["base_types"]
        bt = self.base_type
        thresh = np.empty((self.height, self.width), dtype=np.float32)
        thresh[bt == 0] = base_cfg["bare"]["flooding_threshold"]
        thresh[bt == 1] = base_cfg["sand"]["flooding_threshold"]
        thresh[bt == 2] = base_cfg["soil"]["flooding_threshold"]
        return self.front.ground_water >= thresh