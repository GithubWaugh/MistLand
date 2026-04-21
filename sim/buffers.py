"""
buffers.py
Defines the WorldBuffers dataclass : one numpy array per cell field.
The World object maintains two instances (front / back) for double-buffering.
"""

import numpy as np
from dataclasses import dataclass, field


def _zeros(shape: tuple, dtype) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


@dataclass
class WorldBuffers:
    """
    All mutable per-cell data, stored as numpy arrays of shape (height, width).
    One WorldBuffers instance = one complete snapshot of the world state.
    """

    height: int
    width: int

    # --- Ground layer ---
    ground_water:       np.ndarray = field(init=False)  # float32 [0..+inf)
    nutriments:         np.ndarray = field(init=False)  # uint8   [0..255]
    ground_temp:        np.ndarray = field(init=False)  # float32 (°C)
    albedo:             np.ndarray = field(init=False)  # float32 [0..1]

    # --- Atmospheric layer ---
    pressure:           np.ndarray = field(init=False)  # float32 (bar)
    mist:               np.ndarray = field(init=False)  # uint8   [0..7]
    mist_accumulator:   np.ndarray = field(init=False)  # float32 [0..MIST_UNIT)
    atmo_temp:          np.ndarray = field(init=False)  # float32 (°C)

    # --- Vegetation layer ---
    vegetation:         np.ndarray = field(init=False)  # uint8  [0..4]
    veg_tick_counter:   np.ndarray = field(init=False)  # uint16 (ticks since last change)
    vegetation_water:   np.ndarray = field(init=False)  # float32 water stored in plants

    def __post_init__(self):
        shape = (self.height, self.width)

        self.ground_water       = _zeros(shape, np.float32)
        self.nutriments         = _zeros(shape, np.uint8)
        self.ground_temp        = _zeros(shape, np.float32)
        self.albedo             = _zeros(shape, np.float32)

        self.pressure           = _zeros(shape, np.float32)
        self.mist               = _zeros(shape, np.uint8)
        self.mist_accumulator   = _zeros(shape, np.float32)
        self.atmo_temp          = _zeros(shape, np.float32)

        self.vegetation         = _zeros(shape, np.uint8)
        self.veg_tick_counter   = _zeros(shape, np.uint16)
        self.vegetation_water   = _zeros(shape, np.float32)

    def copy_from(self, other: "WorldBuffers") -> None:
        """Copy all arrays from another WorldBuffers into this one (in-place)."""
        np.copyto(self.ground_water,        other.ground_water)
        np.copyto(self.nutriments,          other.nutriments)
        np.copyto(self.ground_temp,         other.ground_temp)
        np.copyto(self.albedo,              other.albedo)
        np.copyto(self.pressure,            other.pressure)
        np.copyto(self.mist,                other.mist)
        np.copyto(self.mist_accumulator,    other.mist_accumulator)
        np.copyto(self.atmo_temp,           other.atmo_temp)
        np.copyto(self.vegetation,          other.vegetation)
        np.copyto(self.veg_tick_counter,    other.veg_tick_counter)
        np.copyto(self.vegetation_water,    other.vegetation_water)