"""
vegetation.py
Phase 3 : Vegetation growth and devolution.

Vegetation levels (ordered) :
    0 = None
    1 = Lichens  (also displayed as algae when submerged — handled in UI)
    2 = Grass
    3 = Shrubs
    4 = Trees

Fertility modulation :
    world.fertility [-1..1] adjusts thresholds locally.

Desynchronisation :
    Counter reset to random [0, growth_period//2] on grow/devolve.

Flooding rules (applied every tick, immediately) :
    - Flooded cell (ground_water >= flooding_threshold) :
        veg > VEG_LICHENS → forced devolution by one step per tick
        lichen stays as lichen (displayed as algae in UI)
    - Cell adjacent to a flooded cell :
        base_type == BARE → veg_max = VEG_GRASS  (rocky bank)
        base_type == SAND → veg_max = VEG_SHRUBS (sandy bank)
        growth beyond veg_max is blocked

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

from sim.phases.evaporation import MIST_UNIT
from sim.world import VEG_NONE, VEG_LICHENS, VEG_GRASS, VEG_SHRUBS, VEG_TREES
from sim.world import BASE_BARE, BASE_SAND, BASE_SOIL


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

    fertility_water_range = 0.10
    fertility_nut_range   = 5.0
    stress_immunity       = 0.6

    f   = world.front
    b   = world.back
    fer = world.fertility
    bt  = world.base_type

    veg      = f.vegetation.copy()
    gw       = f.ground_water.copy()
    veg_w    = f.vegetation_water.copy()
    nut      = f.nutriments.astype(np.int16)
    counter  = f.veg_tick_counter.copy()

    # --- Random offset for desynchronisation ---
    max_offset = max(1, growth_period // 2)
    rng_offset = np.random.randint(
        0, max_offset, size=counter.shape
    ).astype(np.uint16)

    # --- Flooding masks ---
    flood_thresh = np.empty((world.height, world.width), dtype=np.float32)
    flood_thresh[bt == BASE_BARE] = base_cfg["bare"]["flooding_threshold"]
    flood_thresh[bt == BASE_SAND] = base_cfg["sand"]["flooding_threshold"]
    flood_thresh[bt == BASE_SOIL] = base_cfg["soil"]["flooding_threshold"]

    is_flooded = gw >= flood_thresh

    # Cells adjacent to at least one flooded cell
    has_flooded_neighbour = np.zeros((world.height, world.width), dtype=bool)
    for dr, dc in _NEIGHBOURS:
        has_flooded_neighbour |= np.roll(np.roll(is_flooded, dr, axis=0), dc, axis=1)

    # --- Vegetation cap from bank rules ---
    # Default : no cap (trees allowed)
    veg_max = np.full((world.height, world.width), VEG_TREES, dtype=np.uint8)
    # Rocky bank : max grass
    veg_max[has_flooded_neighbour & (bt == BASE_BARE)] = VEG_GRASS
    # Sandy bank : max shrubs
    veg_max[has_flooded_neighbour & (bt == BASE_SAND)] = VEG_SHRUBS
    # Flooded cells : only lichen survives
    veg_max[is_flooded] = VEG_LICHENS

    # --- Fertility-adjusted thresholds ---
    eff_water_min = (water_min - fer * fertility_water_range).astype(np.float32)
    eff_nut_min   = (nut_min   - fer * fertility_nut_range).astype(np.float32)

    # --- Growth conditions ---
    water_ok  = gw >= eff_water_min
    temp_ok   = (f.ground_temp >= temp_min) & (f.ground_temp <= temp_max)
    nut_ok    = nut.astype(np.float32) >= eff_nut_min
    timer_ok  = counter >= growth_period
    # Lichens et grass peuvent utiliser l'humidité de l'air
    mist_ok  = f.mist >= cfg.get("mist_water_threshold", 3.0)

    can_grow  = water_ok & temp_ok & timer_ok & (veg < veg_max)

    # Lichens : sol OU mist suffisant
    lichen_growth = can_grow & (veg == VEG_NONE) & (water_ok | mist_ok)

    # Grass : sol ET (eau OU mist) — plus exigeant
    normal_growth = can_grow & (veg > VEG_NONE) & nut_ok & (
        water_ok | (mist_ok & (veg == VEG_LICHENS))
    )

    grows = lichen_growth | normal_growth

    # --- Normal devolution conditions ---
    stress   = (~water_ok) | (~temp_ok)
    immune   = fer >= stress_immunity
    devolves = stress & ~immune & (veg > VEG_NONE) & ~grows & ~is_flooded

    # --- Apply growth ---
    # Source d'eau pour la croissance
    using_mist = grows & (veg == VEG_NONE) & ~water_ok & mist_ok

    # Eau consommée depuis le sol (cas normal)
    water_consumed = np.where(
        grows & (veg > VEG_NONE) & ~using_mist,
        cost_water, 0.0
    ).astype(np.float32)
    water_consumed = np.minimum(water_consumed, gw)
    gw    -= water_consumed
    veg_w += water_consumed

    # Eau consommée depuis le mist (lichen depuis humidité)
    mist_consumed = np.where(
        using_mist,
        cost_water / MIST_UNIT,  # convertir en unités mist
        0.0
    ).astype(np.float32)
    mist_consumed = np.minimum(mist_consumed, f.mist)
    # mist est lu depuis front — on écrit dans back
    b.mist = np.clip(f.mist - mist_consumed, 0.0, 7.0).astype(np.float32)
    veg_w += (mist_consumed * MIST_UNIT).astype(np.float32)

    nut = np.where(
        grows & (veg > VEG_NONE),
        nut - cost_nut,
        nut
    ).astype(np.int16)

    veg     = np.where(grows, veg + np.uint8(1), veg).astype(np.uint8)
    counter = np.where(grows, rng_offset, counter).astype(np.uint16)

    # --- Apply normal devolution ---
    gw   += np.where(devolves, veg_w, 0.0).astype(np.float32)
    veg_w = np.where(devolves, 0.0, veg_w).astype(np.float32)
    nut   = np.where(devolves, nut + release_nut, nut).astype(np.int16)
    veg   = np.where(devolves, veg - np.uint8(1), veg).astype(np.uint8)
    counter = np.where(devolves, rng_offset, counter).astype(np.uint16)

    lichen_destroyed = devolves & (f.vegetation == VEG_LICHENS)
    nut = np.where(lichen_destroyed, nut + release_nut, nut).astype(np.int16)

    # --- Apply flooding devolution (one step per tick toward lichen) ---
    flood_devolves = is_flooded & (veg > VEG_LICHENS) & ~grows & ~devolves

    gw   += np.where(flood_devolves, veg_w, 0.0).astype(np.float32)
    veg_w = np.where(flood_devolves, 0.0, veg_w).astype(np.float32)
    nut   = np.where(flood_devolves, nut + release_nut, nut).astype(np.int16)
    veg   = np.where(flood_devolves, veg - np.uint8(1), veg).astype(np.uint8)
    counter = np.where(flood_devolves, rng_offset, counter).astype(np.uint16)

    # --- Increment tick counter ---
    changed = grows | devolves | flood_devolves
    counter = np.where(
        ~changed,
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
    albedo[bt == BASE_BARE] = base_cfg["bare"]["albedo_base"]
    albedo[bt == BASE_SAND] = base_cfg["sand"]["albedo_base"]
    albedo[bt == BASE_SOIL] = base_cfg["soil"]["albedo_base"]
    albedo -= veg.astype(np.float32) * alb_reduction
    albedo  = np.clip(albedo, 0.05, 1.0).astype(np.float32)

    # --- Apply to back buffer ---
    b.vegetation       = veg
    b.ground_water     = gw
    b.vegetation_water = veg_w
    b.nutriments       = nut.astype(np.uint8)
    b.albedo           = albedo
    b.veg_tick_counter = counter