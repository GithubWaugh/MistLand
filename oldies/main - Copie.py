"""
main.py
Entry point for MistLand.
Loads config, generates world, runs N ticks in console,
and prints conservation law checks after each tick.
"""

import json
import time

from sim.world import World
from sim.generation import generate
from sim.phases.evaporation import MIST_UNIT


CONFIG_PATH = "config/default.json"
N_TICKS     = 1000        # set to 1 for phase-by-phase diagnostic
SEED        = 42


def total_water(world: World) -> float:
    return (
        float(world.front.ground_water.sum())
        + float(world.front.mist.sum()) * MIST_UNIT
        + float(world.front.mist_accumulator.sum())
    )


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def water_detail(world):
    gw   = float(world.front.ground_water.sum())
    mist = float(world.front.mist.sum()) * MIST_UNIT
    macc = float(world.front.mist_accumulator.sum())
    vegw = float(world.front.vegetation_water.sum())
    return gw, mist, macc, vegw


def main():
    print("=== MistLand ===")
    print(f"Loading config from '{CONFIG_PATH}'...")
    config = load_config(CONFIG_PATH)

    width  = config["world"]["grid_width"]
    height = config["world"]["grid_height"]
    print(f"Creating world ({width} x {height})...")
    world = World(config)

    print(f"Generating world (seed={SEED})...")
    t0 = time.perf_counter()
    generate(world, seed=SEED)
    t1 = time.perf_counter()
    print(f"Generation done in {t1 - t0:.3f}s")

    water_0  = total_water(world)
    energy_0 = world.total_energy()
    print(f"\nBaseline : water={water_0:.6f}  energy={energy_0:.6f}")
    print("-" * 70)

    for _ in range(N_TICKS):
        w_before = total_water(world)
        world.tick()
        w_after  = total_water(world)
        delta    = w_after - w_before
        if world.tick_count%100 == 0:
            print(f"Tick {world.tick_count:3d} | ΔWater = {delta:+.6f}")
        gw, mist, macc, vegw = water_detail(world)
        if world.tick_count%100 == 0:
            print(f"Tick {world.tick_count:3d} | gw={gw:.2f}  mist={mist:.2f}  macc={macc:.2f}  vegw={vegw:.2f}  total={gw+mist+macc+vegw:.2f}")

    print(f"\nFinal : water={total_water(world):.6f}  energy={world.total_energy():.6f}")
    print("Done.")


if __name__ == "__main__":
    main()