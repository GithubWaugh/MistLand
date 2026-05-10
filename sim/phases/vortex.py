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
        self.evolve_ratio = 0.1
        self.uv_speed = 0.0001
        self.attraction = config["atmosphere"]["vortex_attraction"]

    def random_pos(self):
        self.uv_pos = np.array([np.random.uniform(), np.random.uniform(), 0.0])

    def evolve(self):
        delta = np.array([np.random.normal(), np.random.normal(), 0.0])
        self.uv_dir += delta * self.evolve_ratio
        self.uv_dir /= np.linalg.norm(self.uv_dir)
        self.uv_pos += self.uv_dir * self.uv_speed
        self.uv_pos[:2] %= 1.0

    def world_pos(self, world):
        wx = self.uv_pos[0] * world.width
        wy = self.uv_pos[1] * world.height
        wz = 0.0
        return np.array([wx, wy, wz])
