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
    world_cfg = world.config["world"]
    cfg      = world.config["temperature"]
    base_cfg = world.config["base_types"]

    # Single exchange rate for ground ↔ atmosphere
    # Derived from the average of the two former rates
    ga_rate = cfg["ground_to_atmosphere_rate"]
    ag_rate = cfg["atmosphere_to_ground_rate"]
    exchange_rate = (ga_rate + ag_rate) / 2.0

    gn_rate = cfg["ground_to_neighbour_rate"]

    f = world.front
    b = world.back

    # --- Thermal inertia map ---
    inertia = np.empty((world.height, world.width), dtype=np.float32)
    inertia[world.base_type == 0] = base_cfg["bare"]["thermal_inertia"]
    inertia[world.base_type == 1] = base_cfg["sand"]["thermal_inertia"]
    inertia[world.base_type == 2] = base_cfg["soil"]["thermal_inertia"]
    inertia[world.front.ground_water>1] = 0.05  # Water has high thermal inertia, slows down temperature changes

    # --- Ground ↔ neighbours (conserved by construction) ---
    incoming_ground = np.zeros_like(f.ground_temp)
    total_outgoing  = np.zeros_like(f.ground_temp)
    effective_gn_rate = gn_rate / len(_NEIGHBOURS) # Distribute total exchange rate among neighbours
    for dr, dc in _NEIGHBOURS:
        neighbour_inertia = np.roll(np.roll(inertia, dr, axis=0), dc, axis=1)
        sym_inertia = (inertia + neighbour_inertia) / 2.0
        outgoing = (effective_gn_rate * sym_inertia * f.ground_temp).astype(np.float32)
        incoming_ground += np.roll(np.roll(outgoing, dr, axis=0), dc, axis=1)
        total_outgoing += outgoing

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
    # Sun radiation adds energy to the system, increasing ground temperature
    # or lowering ground temperature if the sun is below the horizon (night).
    solar_input = world_cfg["solar_radiation"]
    sun_rotation_speed = world_cfg.get("sun_rotation_speed", 100)
    sun_phase = 2.0 * np.pi * world.tick_count / sun_rotation_speed
    sun_factor = np.sin(world.uv[:, :, 0] * 2 * np.pi + sun_phase).astype(np.float32)
    artificial_boost = 10.0  # Boost factor to make the sun's effect more visible in the simulation 
    sun_factor *= solar_input**3 * artificial_boost * albedo
        
    # --- Update ---
    b.ground_temp = (f.ground_temp + net_neighbour - net_flux).astype(np.float32)
    b.ground_temp += sun_factor  # Add solar energy input to ground temperature   
    b.atmo_temp   = (f.atmo_temp   + net_flux).astype(np.float32)