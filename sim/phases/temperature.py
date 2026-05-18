"""
temperature.py
Phase 1 : Temperature propagation.

Each tick :
  - Ground temp exchanges heat with its 4 neighbours (toric wrap),
    weighted by thermal inertia.
  - Ground temp and atmosphere temp exchange heat via a single
    NET FLUX — guaranteeing exact energy conservation.

Conservation fix :
  Instead of two independent fluxes (ground→atmo and atmo→ground),
  we compute a single net flux proportional to the temperature
  DIFFERENCE between ground and atmosphere.
  What leaves one layer enters the other exactly — no energy created.

  net_flux = exchange_rate * (ground_temp - atmo_temp)
  ground loses net_flux, atmosphere gains net_flux.
  If atmo is hotter than ground, net_flux is negative → heat flows down.

Neighbour layout (4-connectivity, toric wrap) :
    north = roll(+1, axis=0)
    south = roll(-1, axis=0)
    east  = roll(-1, axis=1)
    west  = roll(+1, axis=1)
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
    world_cfg   = world.config["world"]
    temp_cfg    = world.config["temperature"]
    atmos_cfg   = world.config["atmosphere"]
    base_cfg    = world.config["base_types"]
    veg_cfg     = world.config["plants"]

    # Single exchange rate for ground ↔ atmosphere
    # Derived from the average of the two former rates
    ga_rate = temp_cfg["ground_to_atmosphere_rate"]
    ag_rate = temp_cfg["atmosphere_to_ground_rate"]
    exchange_rate = (ga_rate + ag_rate) / 2.0

    gn_rate = temp_cfg["ground_to_neighbour_rate"]

    f = world.front
    b = world.back

    # --- Thermal inertia map ---
    inertia = np.empty((world.height, world.width), dtype=np.float32)
    inertia[world.base_type == 0] = base_cfg["bare"]["thermal_inertia"]
    inertia[world.base_type == 1] = base_cfg["sand"]["thermal_inertia"]
    inertia[world.base_type == 2] = base_cfg["soil"]["thermal_inertia"]
    inertia[world.front.ground_water>1] = 0.1  # Water has high thermal inertia, slows down temperature changes

    # --- Ground ↔ neighbours (conserved by construction) ---
    incoming_ground = np.zeros_like(f.ground_temp)
    total_outgoing  = np.zeros_like(f.ground_temp)
    effective_gn_rate = gn_rate / len(_NEIGHBOURS) # Distribute total exchange rate among neighbours
    for dr, dc in _NEIGHBOURS:
        neighbour_inertia = np.roll(np.roll(inertia, dr, axis=0), dc, axis=1)
        sym_inertia = (inertia + neighbour_inertia) / 2.0
        outgoing_ground = (effective_gn_rate * sym_inertia * f.ground_temp).astype(np.float32)
        incoming_ground += np.roll(np.roll(outgoing_ground, dr, axis=0), dc, axis=1)
        total_outgoing += outgoing_ground

    net_neighbour = incoming_ground - total_outgoing

    # --- Ground ↔ atmosphere : single net flux ---
    # Positive → heat flows from ground to atmosphere
    # Negative → heat flows from atmosphere to ground
    net_flux = (exchange_rate * (f.ground_temp - f.atmo_temp)).astype(np.float32)

    # Ground albedo reduces solar energy input, keeping ground cooler
        # --- Thermal inertia map ---
    albedo = np.empty((world.height, world.width), dtype=np.float32)
    albedo[world.base_type == 0] = base_cfg["bare"]["albedo_base"]
    albedo[world.base_type == 1] = base_cfg["sand"]["albedo_base"]
    albedo[world.base_type == 2] = base_cfg["soil"]["albedo_base"]
    albedo = np.where(world.front.ground_snow > 0, 0.95, albedo)  # Snow-covered ground is highly reflective (high albedo), keeping it cooler and preserving snow cover longer (positive feedback)
    # Sun radiation adds energy to the system, increasing ground temperature
    # or lowering ground temperature if the sun is below the horizon (night).
    solar_input = world_cfg["solar_radiation"]
    sun_rotation_speed = world_cfg.get("sun_rotation_speed", 100)
    sun_phase = 2.0 * np.pi * float(world.tick_count) / sun_rotation_speed
    sun_factor = np.sin(world.uv[:, :, 0] * 2 * np.pi + sun_phase).astype(np.float32)
    artificial_boost = 1.0  # Boost factor to make the sun's effect more visible in the simulation 
    sun_factor *= solar_input * artificial_boost * (1-albedo)  # Higher solar input has a stronger effect, but is modulated by albedo (reflectivity). Bare ground absorbs more energy, while snow-covered areas reflect most of it.

    # Vegetation reduces Albedo
    veg_albedo_red = np.empty((world.height, world.width), dtype=np.float32)
    veg_albedo_red[world.front.vegetation == 1] = veg_cfg["lichens"]["albedo_reduction"]
    veg_albedo_red[world.front.vegetation == 2] = veg_cfg["shrubs"]["albedo_reduction"]
    veg_albedo_red[world.front.vegetation == 3] = veg_cfg["trees"]["albedo_reduction"]
    albedo = np.minimum(0, albedo - veg_albedo_red)


    # Felt temperature includes mist cooling effect
    mist_cooling_factor = atmos_cfg.get("mist_cooling_factor", 0.0)
    sun_factor -= mist_cooling_factor * world.front.mist.astype(np.float32)/7  # Mist cools the atmosphere by absorbing heat, proportional to mist density and cooling factor

    # --- Update Temperature ---
    b.ground_temp = (f.ground_temp + net_neighbour - net_flux).astype(np.float32)
    b.ground_temp += sun_factor  # Add solar energy input to ground temperature 
    # Felt temperature include altitude cooling effect (lapse rate)
    #lapse_rate = atmos_cfg.get("lapse_rate", 0.0)
    #f.atmo_temp -= lapse_rate * np.maximum(world.surface_altitude(),0.5).astype(np.float32)  # Higher altitudes are cooler, but only count altitudes above a small threshold to prevent excessive cooling in flooded areas
    b.atmo_temp = (f.atmo_temp + net_flux).astype(np.float32)
    b.atmo_temp += sun_factor * 0.5  # Atmosphere also receives some solar energy, but less than the ground (adjust factor as needed)

    
    # Compensate energy loss/gain
    energy_delta = world.total_energy() - world.initial_energy
    if abs(energy_delta) > 10:
        fraction = energy_delta / (world.height * world.width)
        b.ground_temp = np.add(b.ground_temp, -fraction).astype(np.float32)
    