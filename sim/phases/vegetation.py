"""
vegetation.py
Phase 3 : Vegetation growth and devolution.

Vegetation levels (ordered) :
    0 = None
    1 = Lichens
    2 = Grass
    3 = Shrubs
    4 = Trees

Water budget (closed) :
    - On growth   : ground_water -= cost_water
                    vegetation_water += cost_water
    - On devolution : ground_water += vegetation_water (all stored water returned)
                      mist_accumulator += devolution_water_release (extra atmospheric release)
                      vegetation_water = 0

This ensures total water = ground_water + mist + mist_accumulator
                          + vegetation_water  remains constant.

Growth conditions (all must be met) :
    - ground_water >= water_min
    - temp_min <= ground_temp <= temp_max
    - nutriments >= nutriments_min  (except Lichens : grow from nothing)
    - veg_tick_counter >= growth_period_ticks

Devolution conditions (any is sufficient) :
    - ground_water < water_min
    - ground_temp out of [temp_min, temp_max]

Lichens special case :
    - Grow from None with no water/nutriment cost
    - When destroyed (Lichens → None), release extra nutriments
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.world import World

from sim.world import VEG_NONE, VEG_LICHENS, VEG_GRASS, VEG_SHRUBS, VEG_TREES


def step(world: "World") -> None:
    cfg      = world.config["vegetation"]
    base_cfg = world.config["base_types"]
    alb_cfg  = world.config["albedo"]

    growth_period   = cfg["growth_period_ticks"]
    water_min       = cfg["water_min"]
    temp_min        = cfg["temp_min"]
    temp_max        = cfg["temp_max"]
    nut_min         = cfg["nutriments_min"]
    cost_water      = cfg["growth_cost_water"]
    cost_nut        = int(cfg["growth_cost_nutriments"])
    release_nut     = int(cfg["devolution_nutriments_release"])
    release_water   = cfg["devolution_water_release"]
    alb_reduction   = alb_cfg["vegetation_reduction_per_level"]

    f = world.front
    b = world.back

    veg      = f.vegetation.copy()
    gw       = f.ground_water.copy()
    veg_w    = f.vegetation_water.copy()
    nut      = f.nutriments.astype(np.int16)
    macc     = f.mist_accumulator.copy()
    counter  = f.veg_tick_counter.copy()

    # --- Conditions ---
    water_ok  = gw >= water_min
    temp_ok   = (f.ground_temp >= temp_min) & (f.ground_temp <= temp_max)
    nut_ok    = nut >= nut_min
    timer_ok  = counter >= growth_period

    can_grow  = water_ok & temp_ok & timer_ok & (veg < VEG_TREES)

    # Lichens : grow from None without nutriment requirement
    lichen_growth = can_grow & (veg == VEG_NONE)
    normal_growth = can_grow & (veg > VEG_NONE) & nut_ok

    grows    = lichen_growth | normal_growth
    stress   = (~water_ok) | (~temp_ok)
    devolves = stress & (veg > VEG_NONE) & ~grows

    # --- Apply growth ---
    # Water : ground_water → vegetation_water
    water_consumed = np.where(
        grows & (veg > VEG_NONE),   # Lichens have no water cost
        cost_water,
        0.0
    ).astype(np.float32)

    gw    -= water_consumed
    veg_w += water_consumed

    # Nutriments consumed (not Lichens)
    nut = np.where(
        grows & (veg > VEG_NONE),
        nut - cost_nut,
        nut
    ).astype(np.int16)

    veg     = np.where(grows, veg + np.uint8(1), veg).astype(np.uint8)
    counter = np.where(grows, np.uint16(0), counter).astype(np.uint16)

    # --- Apply devolution ---
    # All stored water returned to ground
    gw   += np.where(devolves, veg_w, 0.0).astype(np.float32)
    veg_w = np.where(devolves, 0.0, veg_w).astype(np.float32)

    # Nutriments released
    nut = np.where(devolves, nut + release_nut, nut).astype(np.int16)

    veg     = np.where(devolves, veg - np.uint8(1), veg).astype(np.uint8)
    counter = np.where(devolves, np.uint16(0), counter).astype(np.uint16)

    # Lichens destroyed → extra nutriments
    lichen_destroyed = devolves & (f.vegetation == VEG_LICHENS)
    nut = np.where(lichen_destroyed, nut + release_nut, nut).astype(np.int16)

    # --- Increment tick counter ---
    counter = np.where(
        ~grows & ~devolves,
        np.clip(counter.astype(np.int32) + 1, 0, 65535).astype(np.uint16),
        counter
    ).astype(np.uint16)

    # --- Clamp ---
    veg   = np.clip(veg,   VEG_NONE, VEG_TREES).astype(np.uint8)
    gw    = np.maximum(gw, 0.0).astype(np.float32)
    veg_w = np.maximum(veg_w, 0.0).astype(np.float32)
    nut   = np.clip(nut,  0, 255).astype(np.int16)
    macc  = np.maximum(macc, 0.0).astype(np.float32)

    # --- Update albedo ---
    albedo = np.empty((world.height, world.width), dtype=np.float32)
    albedo[world.base_type == 0] = base_cfg["bare"]["albedo_base"]
    albedo[world.base_type == 1] = base_cfg["sand"]["albedo_base"]
    albedo[world.base_type == 2] = base_cfg["soil"]["albedo_base"]
    albedo -= veg.astype(np.float32) * alb_reduction
    albedo  = np.clip(albedo, 0.05, 1.0).astype(np.float32)

    # --- Apply to back buffer ---
    b.vegetation       = veg
    b.ground_water     = gw
    b.vegetation_water = veg_w
    b.nutriments       = nut.astype(np.uint8)
    b.mist_accumulator = macc
    b.albedo           = albedo
    b.veg_tick_counter = counter