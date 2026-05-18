"""
rain.py
Phase 8 : Rain — atmosphere mist → ground water.

mist is float32 [0..7] — direct conversion, no accumulator.

Conservation : mist lost * mist_to_groundwater == ground_water gained exactly.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.world import World

def step(world: "World") -> None:
    cfg = world.config["rain"]
    mist_to_groundwater = world.config["water"]["mist_to_groundwater"]

    temp_threshold     = cfg["rain_temp_threshold"]
    humidity_threshold = cfg["rain_humidity_threshold"]
    rain_rate          = cfg["rain_rate"]
    saturation_threshold = cfg["rain_saturation_threshold"]

    f = world.front
    b = world.back

    
    # Cells where rain occurs
    is_raining = (
        (f.atmo_temp < temp_threshold) &
        (f.mist >= humidity_threshold)
    )
    is_raining |= (f.mist > saturation_threshold)  # Force rain if mist is maxed out, even if temp is above threshold

    # Mist lost this tick (float)
    mist_loss = np.where(
        is_raining,
        rain_rate * f.mist,
        0.0
    ).astype(np.float32)

    # Clamp to available mist
    mist_loss = np.minimum(mist_loss, f.mist)

    # Water gained = mist lost * mist_to_groundwater
    water_gained = (mist_loss * mist_to_groundwater).astype(np.float32)

    # Apply
    b.ground_water = (f.ground_water + water_gained).astype(np.float32)
    b.mist         = np.clip(f.mist - mist_loss, 0.0, 7.0).astype(np.float32)

    """
    # Rain slighty bends temperature toward 10°C (latent heat release), more so in colder cells
    rain_cooling = np.where(is_raining, 0.02 * (10.0 - f.atmo_temp) * (mist_loss / (rain_rate * humidity_threshold + 1e-6)), 0.0)
    b.atmo_temp = (f.atmo_temp + rain_cooling).astype(np.float32)
    """
    # Rain lowers temperature slightly
    rain_cooling = np.where(is_raining, 0.02 * (mist_loss / (rain_rate * humidity_threshold + 1e-6)), 0.0)
    b.atmo_temp = (f.atmo_temp - rain_cooling).astype(np.float32)