"""
wind.py
Phase 2b : Wind vector update — simplified 2D Navier-Stokes.

The wind field (wind_x, wind_y) is a persistent vector field updated each tick.
Three forces act on the wind :

  1. Pressure gradient  : wind is pushed from high to low pressure
                          new_wind -= k_wind * ∇P

  2. Advection          : the wind carries itself — creates vortices and persistence
                          new_wind -= advection_strength * (wind · ∇)wind

  3. Viscosity          : diffusion, smooths instabilities
                          new_wind += viscosity * ∇²wind

  4. Simplex noise      : spatially coherent, temporally evolving perturbation
                          prevents stagnation without the white-noise jitter of np.random

A damping factor prevents unlimited acceleration.
A max_speed clamp ensures numerical stability.

Convention :
  wind_x  > 0 → eastward  (positive column direction)
  wind_y  > 0 → southward (positive row direction = screen down)

Config keys (all under "atmosphere") :
  k_wind                  : pressure → wind force coefficient
  wind_advection_strength : advection term weight (0 = no memory, 0.3 = rich turbulence)
  wind_viscosity          : diffusion weight
  wind_damping            : multiplicative damping per tick (0.95–0.99)
  wind_max_speed          : speed clamp (prevents divergence)
  wind_noise_strength     : amplitude of the simplex noise perturbation
  wind_noise_frequency    : spatial frequency of the noise (cycles across the world)
  wind_noise_time_scale   : how fast the noise pattern evolves (per tick)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.ndimage import zoom as _zoom

if TYPE_CHECKING:
    from sim.world import World


def step(world: "World") -> None:
    cfg = world.config["atmosphere"]

    k_wind      = cfg.get("k_wind",                    0.1)
    adv_str     = cfg.get("wind_advection_strength",   0.2)
    viscosity   = cfg.get("wind_viscosity",            0.05)
    damping     = cfg.get("wind_damping",              0.98)
    max_speed   = cfg.get("wind_max_speed",            3.0)

    f  = world.front
    b  = world.back
    wx = f.wind_x
    wy = f.wind_y
    P  = f.pressure

    # --- 1. Pressure gradient : wind pushed from high to low pressure ---
    # Central differences (toric wrap)
    dPdx = (np.roll(P, -1, axis=1) - np.roll(P, 1, axis=1)) * 0.5  # eastward ∂P/∂x
    dPdy = (np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)) * 0.5  # southward ∂P/∂y

    # --- 2. Advection : (wind · ∇)wind ---
    dwx_dx = (np.roll(wx, -1, axis=1) - np.roll(wx, 1, axis=1)) * 0.5
    dwx_dy = (np.roll(wx, -1, axis=0) - np.roll(wx, 1, axis=0)) * 0.5
    dwy_dx = (np.roll(wy, -1, axis=1) - np.roll(wy, 1, axis=1)) * 0.5
    dwy_dy = (np.roll(wy, -1, axis=0) - np.roll(wy, 1, axis=0)) * 0.5

    adv_x = -(wx * dwx_dx + wy * dwx_dy)
    adv_y = -(wx * dwy_dx + wy * dwy_dy)

    # --- 3. Viscosity : ν∇²wind (4-neighbour Laplacian) ---
    lap_x = (np.roll(wx,  1, axis=0) + np.roll(wx, -1, axis=0) +
             np.roll(wx,  1, axis=1) + np.roll(wx, -1, axis=1) - 4.0 * wx)
    lap_y = (np.roll(wy,  1, axis=0) + np.roll(wy, -1, axis=0) +
             np.roll(wy,  1, axis=1) + np.roll(wy, -1, axis=1) - 4.0 * wy)


    # --- 4. Update with damping ---
    new_wx = (wx - k_wind * dPdx + adv_str * adv_x + viscosity * lap_x) * damping
    new_wy = (wy - k_wind * dPdy + adv_str * adv_y + viscosity * lap_y) * damping

    # --- 4.5 Coarse O-U noise (spatially coherent, temporally smooth, cheap) ---
    # Maintain a small coarse noise field (H/factor × W/factor) on the world object.
    # Each tick, blend it toward a new white-noise sample (Ornstein-Uhlenbeck process).
    # Upscale with bilinear zoom — total cost: ~256 ops + one bilinear pass.
    noise_strength  = cfg.get("wind_noise_strength",      0.1)
    coarse_factor   = int(cfg.get("wind_noise_coarse_factor", 32))
    persistence     = float(cfg.get("wind_noise_persistence",  0.99))
    cH = max(2, world.height // coarse_factor)
    cW = max(2, world.width  // coarse_factor)
    if world._wind_noise_wx is None:
        world._wind_noise_wx = np.zeros((cH, cW), dtype=np.float32)
    if world._wind_noise_wy is None:    
        world._wind_noise_wy = np.zeros((cH, cW), dtype=np.float32)

    innovation_std = float(np.sqrt(1.0 - persistence ** 2))
    world._wind_noise_wx = (world._wind_noise_wx * persistence
                            + np.random.standard_normal((cH, cW)).astype(np.float32) * innovation_std)
    world._wind_noise_wy = (world._wind_noise_wy * persistence
                            + np.random.standard_normal((cH, cW)).astype(np.float32) * innovation_std)

    zoom_factor = (world.height / cH, world.width / cW)
    new_wx += np.asarray(_zoom(world._wind_noise_wx, zoom_factor, order=1), dtype=np.float32) * noise_strength
    new_wy += np.asarray(_zoom(world._wind_noise_wy, zoom_factor, order=1), dtype=np.float32) * noise_strength

    # --- 5 --- Vortex
    ys = np.arange(world.height, dtype=np.float32)
    xs = np.arange(world.width,  dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    for vtx in world.vortex:
        vx, vy, _ = vtx.world_pos(world)
        dx = (grid_x - vx + world.width  * 0.5) % world.width  - world.width  * 0.5
        dy = (grid_y - vy + world.height * 0.5) % world.height - world.height * 0.5
        dist     = np.maximum(np.sqrt(dx**2 + dy**2), 1e-6)
        max_dist = min(world.width, world.height) / 2.0
        eye_radius = max_dist * cfg.get("vortex_eye_ratio", 0.15)
        t          = np.minimum(dist / eye_radius, 1.0)
        eye_factor = t * t * (3.0 - 2.0 * t)
        falloff    = np.maximum(np.cos(np.pi * 0.5 * dist / max_dist), 0.0) * eye_factor
        new_wx += (dy / dist) * vtx.spin * falloff   # tangential (spin)
        new_wy += (-dx / dist) * vtx.spin * falloff
        new_wx -= (dx / dist) * vtx.pull * falloff   # radial (inward pull)
        new_wy -= (dy / dist) * vtx.pull * falloff

    # Evolve vortices direction for next tick
    for vtx in world.vortex:
        vtx.evolve()


    # --- 6. Speed clamp ---
    speed = np.sqrt(new_wx ** 2 + new_wy ** 2)
    scale = np.where(speed > max_speed, max_speed / (speed + 1e-8), 1.0).astype(np.float32)

    b.wind_x = (new_wx * scale).astype(np.float32)
    b.wind_y = (new_wy * scale).astype(np.float32)
