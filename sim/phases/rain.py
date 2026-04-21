"""
rain.py
Phase 8 : Rain — atmosphere mist → ground water.

Rain occurs when two conditions are met simultaneously :
  - atmo_temp < rain_temp_threshold       (cold enough to condense)
  - mist      >= rain_humidity_threshold  (enough water in the air)

Each tick where it rains, a number of full mist units are converted
to ground water. The number of units actually absorbed by the ground
is computed first (respecting the [0,1] cap), and mist is reduced
by exactly that number — no floating point roundtrip, no water created.

Conservation : mist lost * MIST_UNIT == ground_water gained, exactly.
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

    # --- Cells where rain occurs ---
    is_raining = (
        (f.atmo_temp < temp_threshold) &
        (f.mist >= humidity_threshold)
    )

    # --- Mist units attempting to fall this tick ---
    mist_to_fall = np.where(
        is_raining,
        np.floor(rain_rate * f.mist.astype(np.float32)).astype(np.uint8),
        0
    ).astype(np.uint8)

    # Ensure at least 1 unit falls when raining and mist > 0
    mist_to_fall = np.where(
        is_raining & (f.mist > 0) & (mist_to_fall == 0),
        np.uint8(1),
        mist_to_fall
    ).astype(np.uint8)

    # Clamp : cannot fall more than available mist
    mist_to_fall = np.minimum(mist_to_fall, f.mist)

    # --- How much water would that add to the ground ? ---
    water_to_add = (mist_to_fall.astype(np.float32) * MIST_UNIT)

    # --- How much can the ground actually absorb ? ---
    ground_capacity = (1.0 - f.ground_water).astype(np.float32)  # [0..1]
    water_absorbed  = np.minimum(water_to_add, ground_capacity).astype(np.float32)

    # --- Convert absorbed water back to full mist units ---
    # This is the key : mist lost = floor(water_absorbed / MIST_UNIT)
    # so mist_lost * MIST_UNIT <= water_absorbed exactly.
    mist_lost = np.floor(
        water_absorbed / MIST_UNIT + 1e-6   # small epsilon avoids float undercount
    ).astype(np.uint8)

    # Clamp to mist_to_fall (cannot lose more than what was falling)
    mist_lost = np.minimum(mist_lost, mist_to_fall)

    # --- Actual water added = exactly mist_lost * MIST_UNIT ---
    water_gained = (mist_lost.astype(np.float32) * MIST_UNIT)

    # --- Apply to back buffer ---
    b.ground_water = (f.ground_water + water_gained).astype(np.float32)
    b.mist = (
        f.mist.astype(np.int16) - mist_lost.astype(np.int16)
    ).clip(0, 7).astype(np.uint8)