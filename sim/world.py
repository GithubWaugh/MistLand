"""
world.py
Defines the World class.
"""

import numpy as np
from sim.buffers import WorldBuffers
from sim.phases.vortex import Vortex

VEG_NONE    = 0
VEG_LICHENS = 1
VEG_GRASS   = 2
VEG_SHRUBS  = 3
VEG_TREES   = 4

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

        self.mist_to_groundwater = self.config["water"]["mist_to_groundwater"]

        shape = (self.height, self.width)

        self.altitude   = np.zeros(shape, dtype=np.float32)
        self.base_type  = np.zeros(shape, dtype=np.uint8)
        self.fertility  = np.zeros(shape, dtype=np.float32)
        self.initial_energy = 0.0

        self.uv = np.zeros((self.height, self.width, 2), dtype=np.float32)
        xs = np.linspace(0.0, 1.0, self.width,  endpoint=False, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, self.height, endpoint=False, dtype=np.float32)
        self.uv[:, :, 0] = xs[np.newaxis, :]
        self.uv[:, :, 1] = ys[:, np.newaxis]

        self.front = WorldBuffers(self.height, self.width)
        self.back  = WorldBuffers(self.height, self.width)

        # Coarse noise fields for wind perturbation (wind.py, O-U process).
        # Initialized to None; wind.py allocates them on first tick.
        self._wind_noise_wx: np.ndarray | None = None
        self._wind_noise_wy: np.ndarray | None = None

        self.vortex = np.array([Vortex() for _ in range(self.config["atmosphere"]["vortex_number"])])

    def swap_buffers(self) -> None:
        self.front, self.back = self.back, self.front

    def sync_back_from_front(self) -> None:
        self.back.copy_from(self.front)

    def get_flooded_mask(self) -> np.ndarray:
        """Return boolean mask (height, width) : True where cell is a lake."""
        base_cfg = self.config["base_types"]
        bt       = self.base_type
        thresh   = np.empty((self.height, self.width), dtype=np.float32)
        thresh[bt == BASE_BARE] = base_cfg["bare"]["flooding_threshold"]
        thresh[bt == BASE_SAND] = base_cfg["sand"]["flooding_threshold"]
        thresh[bt == BASE_SOIL] = base_cfg["soil"]["flooding_threshold"]
        return self.front.ground_water >= thresh
    
    def flood_threshold(self) -> np.ndarray:
        base_cfg = self.config["base_types"]
        bt = self.base_type
        thresh = np.zeros(bt.shape, dtype=np.float32)
        thresh[bt == BASE_BARE] = base_cfg["bare"].get("flooding_threshold", 0.0)
        thresh[bt == BASE_SAND] = base_cfg["sand"].get("flooding_threshold", 0.0)
        thresh[bt == BASE_SOIL] = base_cfg["soil"].get("flooding_threshold", 0.0)
        return thresh

    def surface_altitude(self) -> np.ndarray:
        water_to_alt = self.config["water"].get("water_to_altitude")
        water_overhead = np.subtract(self.front.ground_water, self.flood_threshold())
        np.maximum(water_overhead, 0.0, out=water_overhead)  # Only count water above the flood threshold
        return water_overhead * water_to_alt + self.altitude

    def tick(self) -> None:
        from sim.phases import (
            temperature, pressure, vegetation, nutriments,
            evaporation, atmosphere, water, rain, wind, snow
        )
        # Report in Console
        if (self.tick_count%100==0) :
            print(self.report())

        self.sync_back_from_front()
        
        temperature.step(self);  self.swap_buffers(); self.sync_back_from_front()
        pressure.step(self);     self.swap_buffers(); self.sync_back_from_front()
        wind.step(self);         self.swap_buffers(); self.sync_back_from_front()
        vegetation.step(self);   self.swap_buffers(); self.sync_back_from_front()
        nutriments.step(self);   self.swap_buffers(); self.sync_back_from_front()
        evaporation.step(self);  self.swap_buffers(); self.sync_back_from_front()
        snow.step(self);         self.swap_buffers(); self.sync_back_from_front()
        rain.step(self);         self.swap_buffers(); self.sync_back_from_front()  # Rain must be before water to update mist and ground water correctly
        water.step(self);        self.swap_buffers(); self.sync_back_from_front()
        atmosphere.step(self);   self.swap_buffers(); self.sync_back_from_front()
        

        self.tick_count += 1

        for vortex in self.vortex:
            vortex.evolve()

        # NaN guard
        for name, arr in [
            ("pressure",    self.front.pressure),
            ("atmo_temp",   self.front.atmo_temp),
            ("ground_temp", self.front.ground_temp),
            ("mist",        self.front.mist),
            ("wind_x",      self.front.wind_x),
            ("wind_y",      self.front.wind_y),
        ]:
            if np.any(np.isnan(arr)):
                print(f"Tick {self.tick_count} : NaN in {name} !")
                break

    # REPORTS METHODS
    def total_water(self) -> float:
        return (
            float(self.front.ground_water.sum())
            + float(self.front.mist.sum()) * self.mist_to_groundwater
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
