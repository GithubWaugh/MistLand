"""
evaporation.py
Phase 5 : Evaporation — ground water → atmosphere mist.

mist is now float32 [0..7] — no accumulator needed.
Evaporated water is converted directly to mist units (float).

Conservation : total_water = ground_water.sum()
                           + mist.sum() * MIST_UNIT
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.world import World

# One mist unit = this much ground_water
MIST_UNIT = 0.1


def step(world: "World") -> None:
    cfg = world.config["water"]

    temp_threshold = cfg["evap_temp_threshold"]
    evap_rate      = cfg["evap_rate"]

    f = world.front
    b = world.back

    # Cells where evaporation occurs
    can_evaporate = (f.ground_temp > temp_threshold) & (f.ground_water > 0.0)

    # Water evaporating this tick
    evaporated = np.where(
        can_evaporate,
        evap_rate * f.ground_water,
        0.0
    ).astype(np.float32)

    # Clamp to available water
    evaporated = np.minimum(evaporated, f.ground_water)

    # Convert to mist units (float — no rounding)
    mist_gain = (evaporated / MIST_UNIT).astype(np.float32)

    # Clamp mist to [0, 7]
    mist_gain = np.minimum(mist_gain, 7.0 - f.mist)

    # Water actually consumed = mist gained * MIST_UNIT
    water_consumed = (mist_gain * MIST_UNIT).astype(np.float32)

    # Apply
    b.ground_water = (f.ground_water - water_consumed).astype(np.float32)
    b.mist         = np.clip(f.mist + mist_gain, 0.0, 7.0).astype(np.float32)