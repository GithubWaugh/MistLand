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
    # Bare ground : trees cannot grow (applied last, respects stricter caps above)
    veg_max[bt == BASE_BARE] = np.minimum(veg_max[bt == BASE_BARE], VEG_SHRUBS)

    # --- Fertility-adjusted thresholds ---
    eff_water_min = (water_min - fer * fertility_water_range).astype(np.float32)

    # --- Altitude limitation per plant type ---
    plants_cfg = world.config.get("plants", {})
    alt_max_lichens = plants_cfg.get("lichens", {}).get("growth_requirements", {}).get("altitude_max", 1.0)
    alt_max_grass   = plants_cfg.get("grass",   {}).get("growth_requirements", {}).get("altitude_max", 1.0)
    alt_max_shrubs  = plants_cfg.get("shrubs",  {}).get("growth_requirements", {}).get("altitude_max", 1.0)
    alt_max_trees   = plants_cfg.get("trees",   {}).get("growth_requirements", {}).get("altitude_max", 1.0)

    alt = world.altitude
    altitude_ok = (
        ((veg == VEG_NONE)    & (alt <= alt_max_lichens)) |
        ((veg == VEG_LICHENS) & (alt <= alt_max_grass))   |
        ((veg == VEG_GRASS)   & (alt <= alt_max_shrubs))  |
        ((veg == VEG_SHRUBS)  & (alt <= alt_max_trees))
    )

    # --- Temperature limitation per plant type ---
    t_lich = plants_cfg.get("lichens", {}).get("growth_requirements", {}).get("temperature", [temp_min, temp_max])
    t_gras = plants_cfg.get("grass",   {}).get("growth_requirements", {}).get("temperature", [temp_min, temp_max])
    t_shru = plants_cfg.get("shrubs",  {}).get("growth_requirements", {}).get("temperature", [temp_min, temp_max])
    t_tree = plants_cfg.get("trees",   {}).get("growth_requirements", {}).get("temperature", [temp_min, temp_max])

    gt = f.ground_temp
    temp_ok_growth = (
        ((veg == VEG_NONE)    & (gt >= t_lich[0]) & (gt <= t_lich[1])) |
        ((veg == VEG_LICHENS) & (gt >= t_gras[0]) & (gt <= t_gras[1])) |
        ((veg == VEG_GRASS)   & (gt >= t_shru[0]) & (gt <= t_shru[1])) |
        ((veg == VEG_SHRUBS)  & (gt >= t_tree[0]) & (gt <= t_tree[1]))
    )

    # --- Nutriment requirement and cost per plant type ---
    nut_req_lichens = plants_cfg.get("lichens", {}).get("growth_requirements", {}).get("nutriments", nut_min)
    nut_req_grass   = plants_cfg.get("grass",   {}).get("growth_requirements", {}).get("nutriments", nut_min)
    nut_req_shrubs  = plants_cfg.get("shrubs",  {}).get("growth_requirements", {}).get("nutriments", nut_min)
    nut_req_trees   = plants_cfg.get("trees",   {}).get("growth_requirements", {}).get("nutriments", nut_min)

    nut_req = np.select(
        [veg == VEG_NONE, veg == VEG_LICHENS, veg == VEG_GRASS, veg == VEG_SHRUBS],
        [nut_req_lichens,  nut_req_grass,      nut_req_shrubs,   nut_req_trees],
        default=nut_min
    ).astype(np.float32)
    eff_nut_req = (nut_req - fer * fertility_nut_range).astype(np.float32)

    cost_nut_lichens = int(plants_cfg.get("lichens", {}).get("growth_costs", {}).get("nutriments", 0))
    cost_nut_grass   = int(plants_cfg.get("grass",   {}).get("growth_costs", {}).get("nutriments", 0))
    cost_nut_shrubs  = int(plants_cfg.get("shrubs",  {}).get("growth_costs", {}).get("nutriments", cost_nut))
    cost_nut_trees   = int(plants_cfg.get("trees",   {}).get("growth_costs", {}).get("nutriments", cost_nut))

    cost_nut_growth = np.select(
        [veg == VEG_NONE, veg == VEG_LICHENS, veg == VEG_GRASS, veg == VEG_SHRUBS],
        [cost_nut_lichens,  cost_nut_grass,    cost_nut_shrubs,  cost_nut_trees],
        default=0
    ).astype(np.int16)

    # --- Water requirement and cost per plant type ---
    water_req_lichens = plants_cfg.get("lichens", {}).get("growth_requirements", {}).get("water", water_min)
    water_req_grass   = plants_cfg.get("grass",   {}).get("growth_requirements", {}).get("water", water_min)
    water_req_shrubs  = plants_cfg.get("shrubs",  {}).get("growth_requirements", {}).get("water", water_min)
    water_req_trees   = plants_cfg.get("trees",   {}).get("growth_requirements", {}).get("water", water_min)

    water_req = np.select(
        [veg == VEG_NONE, veg == VEG_LICHENS, veg == VEG_GRASS, veg == VEG_SHRUBS],
        [water_req_lichens, water_req_grass, water_req_shrubs, water_req_trees],
        default=water_min
    ).astype(np.float32)
    eff_water_req = (water_req - fer * fertility_water_range).astype(np.float32)

    cost_water_lichens = plants_cfg.get("lichens", {}).get("growth_costs", {}).get("water", cost_water)
    cost_water_grass   = plants_cfg.get("grass",   {}).get("growth_costs", {}).get("water", cost_water)
    cost_water_shrubs  = plants_cfg.get("shrubs",  {}).get("growth_costs", {}).get("water", cost_water)
    cost_water_trees   = plants_cfg.get("trees",   {}).get("growth_costs", {}).get("water", cost_water)

    cost_water_growth = np.select(
        [veg == VEG_NONE, veg == VEG_LICHENS, veg == VEG_GRASS, veg == VEG_SHRUBS],
        [cost_water_lichens, cost_water_grass, cost_water_shrubs, cost_water_trees],
        default=cost_water
    ).astype(np.float32)

    # --- Growth conditions ---
    water_ok        = gw >= eff_water_min    # global — used for devolution stress
    water_ok_growth = gw >= eff_water_req    # per-plant — used for growth gating
    temp_ok   = (f.ground_temp >= temp_min) & (f.ground_temp <= temp_max)
    nut_ok    = nut.astype(np.float32) >= eff_nut_req
    timer_ok  = counter >= growth_period
    # Lichens et grass peuvent utiliser l'humidité de l'air — seuil par type de plante
    _mist_default = cfg.get("mist_water_threshold", 3.0)
    mist_req_lichens = plants_cfg.get("lichens", {}).get("growth_requirements", {}).get("mist", _mist_default)
    mist_req_grass   = plants_cfg.get("grass",   {}).get("growth_requirements", {}).get("mist", _mist_default)
    mist_req_shrubs  = plants_cfg.get("shrubs",  {}).get("growth_requirements", {}).get("mist", _mist_default)
    mist_req_trees   = plants_cfg.get("trees",   {}).get("growth_requirements", {}).get("mist", _mist_default)

    mist_req = np.select(
        [veg == VEG_NONE, veg == VEG_LICHENS, veg == VEG_GRASS, veg == VEG_SHRUBS],
        [mist_req_lichens, mist_req_grass, mist_req_shrubs, mist_req_trees],
        default=_mist_default
    ).astype(np.float32)
    mist_ok = f.mist >= mist_req

    can_grow  = water_ok_growth & temp_ok_growth & timer_ok & (veg < veg_max) & altitude_ok

    # Lichens : sol OU mist suffisant - ajout de nutriment aussi pour les lichens, moins exigeants que les autres
    lichen_growth = can_grow & (veg == VEG_NONE) & (water_ok_growth | mist_ok) & nut_ok

    # Grass : sol ET (eau OU mist) — plus exigeant
    normal_growth = can_grow & (veg > VEG_NONE) & nut_ok & (
        water_ok_growth | (mist_ok & (veg == VEG_LICHENS))
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
        cost_water_growth, 0.0
    ).astype(np.float32)
    water_consumed = np.minimum(water_consumed, gw)
    gw    -= water_consumed
    veg_w += water_consumed

    # Eau consommée depuis le mist (lichen depuis humidité)
    mist_consumed = np.where(
        using_mist,
        cost_water_growth / MIST_UNIT,  # convertir en unités mist
        0.0
    ).astype(np.float32)
    mist_consumed = np.minimum(mist_consumed, f.mist)
    # mist est lu depuis front — on écrit dans back
    b.mist = np.clip(f.mist - mist_consumed, 0.0, 7.0).astype(np.float32)
    veg_w += (mist_consumed * MIST_UNIT).astype(np.float32)

    nut = np.where(grows, nut - cost_nut_growth, nut).astype(np.int16)

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
    alb_red_lichens = plants_cfg.get("lichens", {}).get("albedo_reduction", alb_reduction)
    alb_red_grass   = plants_cfg.get("grass",   {}).get("albedo_reduction", alb_reduction)
    alb_red_shrubs  = plants_cfg.get("shrubs",  {}).get("albedo_reduction", alb_reduction)
    alb_red_trees   = plants_cfg.get("trees",   {}).get("albedo_reduction", alb_reduction)

    alb_cum = np.array([
        0.0,
        alb_red_lichens,
        alb_red_lichens + alb_red_grass,
        alb_red_lichens + alb_red_grass + alb_red_shrubs,
        alb_red_lichens + alb_red_grass + alb_red_shrubs + alb_red_trees,
    ], dtype=np.float32)

    albedo -= alb_cum[veg]
    albedo  = np.clip(albedo, 0.05, 1.0).astype(np.float32)

    # --- Apply to back buffer ---
    b.vegetation       = veg
    b.ground_water     = gw
    b.vegetation_water = veg_w
    b.nutriments       = nut.astype(np.uint8)
    b.albedo           = albedo
    b.veg_tick_counter = counter