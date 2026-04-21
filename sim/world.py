"""
world.py
Defines the World class : holds all static and mutable world data,
manages the double buffer, and drives the tick loop.
"""

import numpy as np
from sim.buffers import WorldBuffers
from sim.phases.evaporation import MIST_UNIT

# Vegetation levels — used as readable constants throughout the codebase
VEG_NONE    = 0
VEG_LICHENS = 1
VEG_GRASS   = 2
VEG_SHRUBS  = 3
VEG_TREES   = 4

# Base types
BASE_BARE   = 0
BASE_SAND   = 1
BASE_SOIL   = 2


class World:
    """
    Central object for the simulation.

    Static data (never modified after generation) :
        altitude, base_type, uv

    Mutable data (updated every tick) :
        Stored in two WorldBuffers instances (front / back).
        The simulation writes to `back`, the renderer reads from `front`.
        After each tick, front and back are swapped.
    """

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

        # UV : normalized (u, v) coordinates for each cell
        # uv[y, x, 0] = u = x / width   (horizontal, 0..1)
        # uv[y, x, 1] = v = y / height  (vertical,   0..1)
        self.uv = np.zeros((self.height, self.width, 2), dtype=np.float32)
        xs = np.linspace(0.0, 1.0, self.width,  endpoint=False, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, self.height, endpoint=False, dtype=np.float32)
        self.uv[:, :, 0] = xs[np.newaxis, :]   # u varies along x
        self.uv[:, :, 1] = ys[:, np.newaxis]   # v varies along y

        # --- Double buffer ---
        self.front  = WorldBuffers(self.height, self.width)
        self.back   = WorldBuffers(self.height, self.width)

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def swap_buffers(self) -> None:
        """Swap front and back buffers (pointer swap, no data copy)."""
        self.front, self.back = self.back, self.front

    def sync_back_from_front(self) -> None:
        """
        Copy front into back at the start of a tick, so that phases
        which only partially update the world start from a consistent state.
        """
        self.back.copy_from(self.front)

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Conservation checks (for validation during development)
    # ------------------------------------------------------------------

    def total_water(self) -> float:
        """Sum of all water in the world (ground + atmosphere). Should be constant."""
        ground  = float(self.front.ground_water.sum())
        atmo    = (float(self.front.mist.sum()) * MIST_UNIT
                 + float(self.front.mist_accumulator.sum()))
        return ground + atmo

    def total_energy(self) -> float:
        """Sum of all temperatures in the world. Should be approximately constant."""
        ground  = float(self.front.ground_temp.sum())
        atmo    = float(self.front.atmo_temp.sum())
        return ground + atmo

    def report(self) -> str:
        """Return a one-line summary of the current world state."""
        return (
            f"Tick {self.tick_count:6d} | "
            f"Water {self.total_water():.2f} | "
            f"Energy {self.total_energy():.2f}"
        )