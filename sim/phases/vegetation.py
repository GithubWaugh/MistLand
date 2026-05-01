"""
vegetation.py
Phase 3 : Vegetation growth and devolution.

Vegetation levels (ordered) :
    0 = None
    1 = Lichens
    2 = Grass
    3 = Shrubs
    4 = Trees

Fertility modulation :
    world.fertility [-1..1] adjusts thresholds locally :
    - Positive fertility : lower water/nutriment requirements,
                          higher stress tolerance (immune to devolution)
    - Negative fertility : higher requirements, lower stress tolerance

    effective_water_min  = water_min  - fertility * fertility_water_range
    effective_nut_min    = nut_min    - fertility * fertility_nut_range
    devolution blocked if fertility > stress_immunity

Desynchronisation :
    When a cell grows or devolves, its tick counter is reset to a random
    value in [0, growth_period // 2] instead of 0. This breaks the
    synchronous wave behaviour seen in large homogeneous patches.

Water budget (closed) :
    - On growth   : ground_water -= cost_water (clamped to available)
                    vegetation_water += cost_water
    - On devolution : ground_water += vegetation_water
                      vegetation_water = 0
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
    alb_reduction   = alb_cfg["vegetation_reduction_per_level"]

    # Fertility modulation ranges
    fertility_water_range = 0.10
    fertility_nut_range   = 5.0
    stress_immunity       = 0.6

    f   = world.front
    b   = world.back
    fer = world.fertility   # float32 [-1..1], static

    veg      = f.vegetation.copy()
    gw       = f.ground_water.copy()
    veg_w    = f.vegetation_water.copy()
    nut      = f.nutriments.astype(np.int16)
    counter  = f.veg_tick_counter.copy()

    # --- Random offset for counter reset (desynchronisation) ---
    # Each cell that grows or devolves restarts from a random point
    # in [0, growth_period // 2] rather than 0, breaking patch synchrony.
    max_offset = max(1, growth_period // 2)
    rng_offset = np.random.randint(
        0, max_offset,
        size=counter.shape
    ).astype(np.uint16)

    # --- Fertility-adjusted thresholds ---
    eff_water_min = (water_min - fer * fertility_water_range).astype(np.float32)
    eff_nut_min   = (nut_min   - fer * fertility_nut_range).astype(np.float32)

    # --- Growth conditions ---
    water_ok  = gw >= eff_water_min
    temp_ok   = (f.ground_temp >= temp_min) & (f.ground_temp <= temp_max)
    nut_ok    = nut.astype(np.float32) >= eff_nut_min
    timer_ok  = counter >= growth_period

    can_grow  = water_ok & temp_ok & timer_ok & (veg < VEG_TREES)

    lichen_growth = can_grow & (veg == VEG_NONE)
    normal_growth = can_grow & (veg > VEG_NONE) & nut_ok

    grows = lichen_growth | normal_growth

    # --- Devolution conditions ---
    stress   = (~water_ok) | (~temp_ok)
    immune   = fer >= stress_immunity
    devolves = stress & ~immune & (veg > VEG_NONE) & ~grows

    # --- Apply growth ---
    water_consumed = np.where(
        grows & (veg > VEG_NONE),
        cost_water,
        0.0
    ).astype(np.float32)

    # Clamp : cannot consume more than available
    water_consumed = np.minimum(water_consumed, gw)

    gw    -= water_consumed
    veg_w += water_consumed

    nut = np.where(
        grows & (veg > VEG_NONE),
        nut - cost_nut,
        nut
    ).astype(np.int16)

    veg     = np.where(grows, veg + np.uint8(1), veg).astype(np.uint8)
    counter = np.where(grows, rng_offset, counter).astype(np.uint16)

    # --- Apply devolution ---
    gw   += np.where(devolves, veg_w, 0.0).astype(np.float32)
    veg_w = np.where(devolves, 0.0, veg_w).astype(np.float32)

    nut = np.where(devolves, nut + release_nut, nut).astype(np.int16)

    veg     = np.where(devolves, veg - np.uint8(1), veg).astype(np.uint8)
    counter = np.where(devolves, rng_offset, counter).astype(np.uint16)

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
    gw    = np.maximum(gw,   0.0).astype(np.float32)
    veg_w = np.maximum(veg_w, 0.0).astype(np.float32)
    nut   = np.clip(nut,   0, 255).astype(np.int16)

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
    b.albedo           = albedo
    b.veg_tick_counter = counter