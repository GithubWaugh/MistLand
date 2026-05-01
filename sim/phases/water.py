"""
water.py
Phase 7 : Water movements — runoff and flooding.

Each tick :
  - Runoff is driven by HYDRAULIC ALTITUDE rather than terrain altitude alone :
      effective_alt(cell) = altitude(cell) + ground_water(cell) * water_to_altitude
    This prevents water from flowing uphill into already-flooded cells.
    A cell with gw=18 at altitude 0.3 has a higher water surface than a
    dry neighbour at the same terrain altitude — water stays put.
  - Sand and soil retain a minimum amount of water (retention_min).
  - Flooded cells (lakes) generate nutriments for neighbours.
    Runoff from lakes also carries nutriments downstream.

Conservation : total ground_water remains constant.
  ground_water is NOT capped at 1.0 — values above flooding_threshold
  indicate a lake cell.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.world import World


_NEIGHBOURS = [
    (-1,  0),  # north
    ( 1,  0),  # south
    ( 0,  1),  # east
    ( 0, -1),  # west
    (-1,  1),  # NE
    (-1, -1),  # NW
    ( 1,  1),  # SE
    ( 1, -1),  # SW
]


def step(world: "World") -> None:
    cfg      = world.config["water"]
    base_cfg = world.config["base_types"]
    nut_cfg  = world.config["nutriments"]

    runoff_rate   = cfg["runoff_rate"]
    water_to_alt  = cfg.get("water_to_altitude", 0.1)

    f = world.front
    b = world.back

    alt = world.altitude
    bt  = world.base_type

    # --- Retention minimum per cell ---
    retention = np.empty_like(alt)
    retention[bt == 0] = base_cfg["bare"]["water_retention"]
    retention[bt == 1] = base_cfg["sand"]["water_retention"]
    retention[bt == 2] = base_cfg["soil"]["water_retention"]

    # --- Flooding threshold per cell ---
    flood_thresh = np.empty_like(alt)
    flood_thresh[bt == 0] = base_cfg["bare"]["flooding_threshold"]
    flood_thresh[bt == 1] = base_cfg["sand"]["flooding_threshold"]
    flood_thresh[bt == 2] = base_cfg["soil"]["flooding_threshold"]

    # --- Hydraulic altitude : terrain + water column ---
    # Water only flows from cells where the water SURFACE is higher,
    # not just where the terrain is higher.
    effective_alt = (alt + f.ground_water * water_to_alt).astype(np.float32)

    # --- Hydraulic altitude deltas to each neighbour ---
    deltas = []
    for dr, dc in _NEIGHBOURS:
        shifted = np.roll(np.roll(effective_alt, dr, axis=0), dc, axis=1)
        d = effective_alt - shifted
        deltas.append(np.maximum(d, 0.0).astype(np.float32))

    total_delta = np.add.reduce(deltas).astype(np.float32)

    # --- Water available for runoff (above retention minimum) ---
    available = np.maximum(f.ground_water - retention, 0.0).astype(np.float32)

    # Total outflow : only where downhill neighbours exist
    outflow = np.where(
        total_delta > 0.0,
        runoff_rate * available,
        0.0
    ).astype(np.float32)

    # Clamp : cannot send more than available
    outflow = np.minimum(outflow, available)

    # --- Build new ground water : start from current, subtract outflow ---
    new_ground_water = (f.ground_water - outflow).astype(np.float32)

    # --- Add inflow from each neighbour ---
    for i, (dr, dc) in enumerate(_NEIGHBOURS):
        fraction = np.where(
            total_delta > 0.0,
            deltas[i] / (total_delta + 1e-8),
            0.0
        ).astype(np.float32)

        water_to_neighbour = (outflow * fraction).astype(np.float32)
        new_ground_water  += np.roll(np.roll(water_to_neighbour, -dr, axis=0), -dc, axis=1)

    new_ground_water = np.maximum(new_ground_water, 0.0).astype(np.float32)

    # --- Flooding : cells above threshold are lakes ---
    is_lake = new_ground_water >= flood_thresh

    # Lakes generate nutriments for neighbours
    lake_rate      = int(nut_cfg["lake_generation_rate"])
    new_nutriments = f.nutriments.copy()

    for axis, shift in _NEIGHBOURS:
        neighbour_is_lake = np.roll(is_lake, -shift, axis=axis)
        new_nutriments = np.clip(
            new_nutriments.astype(np.int16) + np.where(neighbour_is_lake, lake_rate, 0),
            0, 255
        ).astype(np.uint8)

    # Runoff from lakes carries nutriments downstream
    nut_transport = nut_cfg["runoff_transport_rate"]
    lake_outflow  = np.where(is_lake, outflow, 0.0).astype(np.float32)

    for i, (axis, shift) in enumerate(_NEIGHBOURS):
        fraction = np.where(
            total_delta > 0.0,
            deltas[i] / (total_delta + 1e-8),
            0.0
        ).astype(np.float32)

        nut_carried = (lake_outflow * fraction * nut_transport * 255).astype(np.int16)
        new_nutriments = np.clip(
            new_nutriments.astype(np.int16)
            + np.roll(nut_carried, -shift, axis=axis),
            0, 255
        ).astype(np.uint8)

    # --- Force conservation (float32 rounding correction) ---
    water_before = f.ground_water.sum()
    water_after  = new_ground_water.sum()
    error        = water_before - water_after
    if abs(error) > 1e-6:
        new_ground_water += error / (world.height * world.width)

    # --- Apply to back buffer ---
    b.ground_water = new_ground_water
    b.nutriments   = new_nutriments