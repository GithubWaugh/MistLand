"""
evaporation.py
Phase 5 : Evaporation — ground water → atmosphere mist.

mist is now float32 [0..7] — no accumulator needed.
Evaporated water is converted directly to mist units (float).

Conservation : total_water = ground_water.sum()
                           + mist.sum() * mist_to_groundwater
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.world import World

def step(world: "World") -> None:
    cfg = world.config["water"]

    temp_threshold = cfg["evap_temp_threshold"]
    water_temp_threshold = cfg["evap_water_temp_threshold"]
    evap_rate      = cfg["evap_rate"]
    # One mist unit to ground_water conversion rate 
    mist_to_groundwater = cfg["mist_to_groundwater"]

    f = world.front
    b = world.back

    # Cells where evaporation occurs
    can_evaporate = (f.ground_temp > temp_threshold) & (f.ground_water > 0.0)

    # Flooded cells (ground_water >= type-specific threshold) evaporate only when temp exceeds water threshold
    bt_cfg = world.config["base_types"]
    flood_thresholds = np.array([
        bt_cfg["bare"]["flooding_threshold"],
        bt_cfg["sand"]["flooding_threshold"],
        bt_cfg["soil"]["flooding_threshold"],
    ], dtype=np.float32)
    cell_flood_threshold = flood_thresholds[world.base_type]

    # Water evaporating this tick
    evaporated = np.where(
    can_evaporate,
    evap_rate * np.minimum(f.ground_water-cell_flood_threshold, 1.0),  # ← cap à 1.0
    0.0 ).astype(np.float32)

    # Clamp to available water
    evaporated = np.minimum(evaporated, f.ground_water)

    # Convert to mist units (float — no rounding)
    mist_gain = (evaporated / mist_to_groundwater).astype(np.float32)

    # Clamp mist to [0, 7]
    mist_gain = np.minimum(mist_gain, 7.0 - f.mist)

    # Water actually consumed = mist gained * mist_to_groundwater
    water_consumed = (mist_gain * mist_to_groundwater).astype(np.float32)

    # Apply
    b.ground_water = (f.ground_water - water_consumed).astype(np.float32)
    b.mist         = np.clip(f.mist + mist_gain, 0.0, 7.0).astype(np.float32)