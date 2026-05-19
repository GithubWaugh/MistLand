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
    # Water flows from cells where the water SURFACE is higher,
    # not just where the terrain is higher.

    # --- Hydraulic altitude deltas to each neighbour ---
    deltas = []
    for dr, dc in _NEIGHBOURS:
        shifted = np.roll(np.roll(world.surface_altitude(), dr, axis=0), dc, axis=1)
        d = world.surface_altitude() - shifted
        deltas.append(np.maximum(d, 0.0).astype(np.float32))

    total_delta = np.add.reduce(deltas).astype(np.float32)

    # --- Water available for runoff (above retention minimum) ---
    available = np.where(~f.lake, f.ground_water,np.maximum(f.ground_water - retention, 0.0).astype(np.float32))

    # Total outflow : only where downhill neighbours exist.
    # Cap to half the surface gap (in water units) so we never overshoot equilibrium.
    max_equalize = (total_delta / water_to_alt * 0.5).astype(np.float32)
    outflow = np.where(
        total_delta > 0.0,
        np.minimum(runoff_rate * available, max_equalize),
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


    # --- Flooding : cells above threshold are lakes ---
    is_lake = world.get_flooded_mask()
    # When a cell becomes newly flooded, apply an initial subsidence to create a lake basin and prevent immediate overspill.
    newly_flooded = is_lake & ~world.front.lake
    if newly_flooded.any():
        # apply a small subsidence only where new lakes form
        subsidence = (0.01 * newly_flooded.astype(np.float32))
        world.altitude = np.maximum(world.altitude - subsidence, 0.0).astype(np.float32) # Initial subsidence to create a lake basin
    # When a cell emerges from flooding, raise the terrain slightly to prevent it from immediately reflooding.
    newly_emerged = ~is_lake & world.front.lake
    if newly_emerged.any():
        emergence_rise = (0.01 * newly_emerged.astype(np.float32))
        world.altitude = (world.altitude + emergence_rise).astype(np.float32) # Raise terrain slightly to prevent immediate reflooding

    # Erode terrain under water flow : 0.1% of outflow volume, capped to 0.01 altitude units per tick
    erosion_rate = cfg.get("erosion_rate", 0.001)
    erosion = np.maximum(outflow * erosion_rate, 0).astype(np.float32)
    erosion /= world.front.vegetation + 1 # Vegetation reduces erosion
    erosion *= ~is_lake # No erosion in flooded cells (lakes) to prevent excessive deepening
    erosion /= world.base_type + 1  # Rock doesn't erode, Sand erodes less than soil
    world.altitude = np.maximum(world.altitude - erosion, 0.0).astype(np.float32)
    
    # Clamp : no negative water quantity
    new_ground_water = np.maximum(new_ground_water, 0.0).astype(np.float32)

    # Flooded snow melts into water, adding to the lake's ground water
    snow_melt = np.where(is_lake, f.ground_snow, 0.0).astype(np.float32)
    new_ground_water += snow_melt
    b.ground_snow -= snow_melt

    # Lakes generate nutriments for neighbours
    lake_rate      = int(nut_cfg["lake_generation_rate"])
    new_nutriments = f.nutriments.copy()

    for dr, dc in _NEIGHBOURS:
        neighbour_is_lake = np.roll(np.roll(is_lake, dr, axis=0), dc, axis=1)
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

    
    # --- Ground water diffusion for non-lake cells ---
    # Non-lake cells spread 5% of their ground water equally to 8 neighbours.
    gw_diffusion_rate = cfg.get("ground_water_diffusion_rate", 0.05)
    diffuse_out = np.where(~is_lake, gw_diffusion_rate * new_ground_water, 0.0).astype(np.float32)
    diffuse_per_neighbour = (diffuse_out / 8.0).astype(np.float32)

    new_ground_water -= diffuse_out
    for dr, dc in _NEIGHBOURS:
        new_ground_water += np.roll(np.roll(diffuse_per_neighbour, -dr, axis=0), -dc, axis=1)

    new_ground_water = np.maximum(new_ground_water, 0.0).astype(np.float32)


    # Frozen grounds (tundra) : ground water turns to snow, ground water set to zero
    frozen = (f.ground_temp < 0.0) & ~is_lake
    b.ground_snow = (b.ground_snow + np.where(frozen, new_ground_water, 0.0)).astype(np.float32)
    new_ground_water = np.where(frozen, 0.0, new_ground_water).astype(np.float32)

    # Compensate water loss/gain — correction distributed over lake cells only
    if (world.tick_count % 10 == 0):  # Only apply correction every 10 ticks to avoid over-correction and oscillations
        water_goal  = world.height * world.width * world.config["world"]["total_water"]
        water_delta = float(new_ground_water.sum()) - water_goal
        lake_count  = float(is_lake.sum())
        if abs(water_delta) > 10 and lake_count > 0:
            correction = -water_delta / lake_count
            new_ground_water = np.where(is_lake, new_ground_water + correction, new_ground_water).astype(np.float32)

    # --- Lake surface leveling: damp oscillations within lake bodies ---
    # Nudge each lake cell toward the local mean of its lake-only neighbours.
    # Non-lake neighbours are excluded so no water leaks across the lake boundary.
    # A rescale step enforces exact conservation of total lake water.
    lake_smooth_rate = cfg.get("lake_smooth_rate", 0.4)
    if is_lake.any() and lake_smooth_rate > 0.0:
        gw_sum  = new_ground_water.copy()
        n_count = np.ones_like(new_ground_water)
        for dr, dc in _NEIGHBOURS:
            nb_gw   = np.roll(np.roll(new_ground_water, dr, axis=0), dc, axis=1)
            nb_lake = np.roll(np.roll(is_lake.astype(np.float32), dr, axis=0), dc, axis=1)
            gw_sum  += nb_gw   * nb_lake
            n_count += nb_lake
        local_mean   = gw_sum / n_count
        smoothed     = new_ground_water + lake_smooth_rate * (local_mean - new_ground_water)
        total_before = float(new_ground_water[is_lake].sum())
        new_ground_water = np.where(is_lake, smoothed, new_ground_water).astype(np.float32)
        total_after  = float(new_ground_water[is_lake].sum())
        if total_after > 1e-6:
            new_ground_water[is_lake] *= total_before / total_after

    # --- Apply to back buffer ---
    b.ground_water = new_ground_water
    b.nutriments   = new_nutriments
    b.lake         = is_lake