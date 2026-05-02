"""
rain.py
Phase 8 : Rain — atmosphere mist → ground water.

mist is float32 [0..7] — direct conversion, no accumulator.

Conservation : mist lost * MIST_UNIT == ground_water gained exactly.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.world import World

from sim.phases.evaporation import MIST_UNIT


def step(world: "World") -> None:
    cfg = world.config["rain"]

    temp_threshold     = cfg["rain_temp_threshold"]
    humidity_threshold = cfg["rain_humidity_threshold"]
    rain_rate          = cfg["rain_rate"]

    f = world.front
    b = world.back

    is_flooded = world.get_flooded_mask()  # precompute flooded mask for efficiency

    # Cells where rain occurs
    is_raining = (
        (f.atmo_temp < temp_threshold) &
        (f.mist >= humidity_threshold)
    )
    # Abandon de  & ~is_flooded   # pas de pluie sur les lacs déjà saturés -> trop artificiel

    # Mist lost this tick (float)
    mist_loss = np.where(
        is_raining,
        rain_rate * f.mist,
        0.0
    ).astype(np.float32)

    # Clamp to available mist
    mist_loss = np.minimum(mist_loss, f.mist)

    # Water gained = mist lost * MIST_UNIT
    water_gained = (mist_loss * MIST_UNIT).astype(np.float32)

    # Clamp to ground capacity
    ground_capacity = np.maximum(0.0, 1.0 - f.ground_water)
    water_absorbed  = np.minimum(water_gained, ground_capacity).astype(np.float32)

    # Mist actually consumed = water absorbed / MIST_UNIT
    mist_actually_lost = (water_absorbed / MIST_UNIT).astype(np.float32)

    # Apply
    b.ground_water = (f.ground_water + water_absorbed).astype(np.float32)
    b.mist         = np.clip(f.mist - mist_actually_lost, 0.0, 7.0).astype(np.float32)