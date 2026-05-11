"""
Vortex

"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.world import World

class Vortex:
    def __init__(self, config: dict) -> None:
        self.uv_pos = np.array([0.0, 0.0, 0.0])
        d = np.array([np.random.normal(), np.random.normal(), 0.0])
        self.uv_dir = d / np.linalg.norm(d)
        self.evolve_ratio = 0.05
        self.uv_speed = 0.0005
        self.spin = config["atmosphere"]["vortex_spin"]
        self.pull = config["atmosphere"]["vortex_pull"]

    def random_pos(self):
        self.uv_pos = np.array([np.random.uniform(), np.random.uniform(), 0.0])

    def evolve(self):
        delta = np.array([np.random.normal(), np.random.normal(), 0.0])
        self.uv_dir += delta * self.evolve_ratio
        self.uv_dir /= np.linalg.norm(self.uv_dir)
        self.uv_pos += self.uv_dir * self.uv_speed
        self.uv_pos[:2] %= 1.0

    def stick_to_equator(self):
        # Optional: add a weak attraction to the equator (y=0.5) to keep the vortex more dynamic and prevent it from drifting too much towards the poles
        equator_attraction = 2 * (0.5 - self.uv_pos[1])
        self.uv_dir[1] += equator_attraction * 2.5  # Stronger attraction to the equator
        self.uv_dir = self.uv_dir / np.linalg.norm(self.uv_dir)

    def world_pos(self, world):
        wx = self.uv_pos[0] * world.width
        wy = self.uv_pos[1] * world.height
        wz = 0.0
        return np.array([wx, wy, wz])
