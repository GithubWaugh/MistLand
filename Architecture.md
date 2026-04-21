# Architecture Notes

## Project overview

Python-based world simulation. The backend computes the world state tick by tick in RAM using numpy arrays. The frontend (pygame) reads the current state and renders it. Both run in the same Python process, in separate threads.

---

## Repository structure

```
MistLand/
│
├── config/
│   └── default.json          # All tunable parameters (see Concept.md § Configuration reference)
│
├── data/
│   └── saves/                # World state dumps (manual save/load)
│
├── sim/                      # Backend — simulation logic
│   ├── __init__.py
│   ├── world.py              # World class : holds all numpy arrays, exposes tick()
│   ├── generation.py         # World generation : simplex noise, initial state
│   ├── phases/               # One module per simulation phase
│   │   ├── __init__.py
│   │   ├── temperature.py
│   │   ├── pressure.py
│   │   ├── vegetation.py
│   │   ├── nutriments.py
│   │   ├── evaporation.py
│   │   ├── atmosphere.py
│   │   ├── water.py
│   │   └── rain.py
│   └── io.py                 # Save / load world state to disk
│
├── ui/                       # Frontend — rendering and input
│   ├── __init__.py
│   ├── app.py                # Main pygame loop, thread management
│   ├── map_view.py           # 2D top-down grid view
│   ├── overlays.py           # Temperature, pressure, water overlays
│   ├── text_frame.py         # Command-line interaction panel
│   └── torus_view.py         # 3D torus view (ModernGL) — later milestone
│
├── main.py                   # Entry point
├── requirements.txt
└── README.md
```

---

## Data structures

### World state — numpy arrays

All world data is stored as numpy arrays of shape `(height, width)`, i.e. `[y, x]`.
All arrays live in a single `World` object.

#### Static arrays *(set at generation, never modified during simulation)*

| Array | dtype | Description |
|-------|-------|-------------|
| `altitude` | `float32` | Cell altitude, generated via simplex noise |
| `base_type` | `uint8` | 0 = bare, 1 = sand, 2 = soil |

#### Ground layer *(updated each tick)*

| Array | dtype | Range | Description |
|-------|-------|-------|-------------|
| `ground_water` | `float32` | [0..1] | Water quantity at ground level |
| `nutriments` | `uint8` | [0..255] | Organic nutriment quantity |
| `ground_temp` | `float32` | °C | Ground temperature |
| `albedo` | `float32` | [0..1] | Energy reflection coefficient |

#### Atmospheric layer *(updated each tick)*

| Array | dtype | Range | Description |
|-------|-------|-------|-------------|
| `pressure` | `float32` | bar | Atmospheric pressure (drives wind) |
| `mist` | `uint8` | [0..7] | Airborne water quantity |
| `atmo_temp` | `float32` | °C | Atmospheric temperature |

#### Vegetation layer *(updated each tick)*

| Array | dtype | Range | Description |
|-------|-------|-------|-------------|
| `vegetation` | `uint8` | [0..4] | 0=None, 1=Lichens, 2=Grass, 3=Shrubs, 4=Trees |
| `veg_tick_counter` | `uint16` | ticks | Ticks since last vegetation change (for growth period) |

#### UV coordinates *(static, set at generation)*

| Array | dtype | Range | Description |
|-------|-------|-------|-------------|
| `uv` | `float32` | [0..1] | Normalized (u, v) coordinates, shape (height, width, 2) |

---

### Double buffer

To avoid race conditions between the simulation thread and the render thread, the `World` object maintains two sets of mutable arrays : **buffer A** and **buffer B**.

- The simulation writes to the **back buffer**
- The renderer reads from the **front buffer**
- After each tick, buffers are swapped (pointer swap, no data copy)

```python
# Simplified swap
world.front, world.back = world.back, world.front
```

---

## Tick loop

Each tick executes the simulation phases in order, all operating on numpy arrays :

```python
def tick(world: World) -> None:
    temperature.step(world)
    pressure.step(world)
    vegetation.step(world)
    nutriments.step(world)
    evaporation.step(world)
    atmosphere.step(world)
    water.step(world)
    rain.step(world)
    world.swap_buffers()
    world.tick_count += 1
```

Each phase module exposes a single `step(world)` function. Phases read from `world.back` and write to `world.back` (the front buffer is untouched during computation).

### Tick loop — note on buffer swap

Each phase is followed by swap_buffers() + sync_back_from_front(),
so that every phase reads a fully consistent and up-to-date world state.
This ensures inter-phase dependencies are respected within a single tick
(e.g. evaporation modifies mist before rain reads it).

---

## Threading model

```
Main thread
  └── pygame event loop + rendering (reads world.front)

Simulation thread
  └── tick loop (writes world.back, then swaps)
```

Synchronization : a single `threading.Event` flag.

```python
tick_done = threading.Event()

# Simulation thread
while running:
    if auto_mode or step_requested:
        tick(world)
        tick_done.set()         # signal renderer
        step_requested = False

# Render thread
while running:
    tick_done.wait()
    tick_done.clear()
    render(world.front)         # safe read
```

---

## Save / load

World state is saved as a collection of numpy `.npy` files, one per array, bundled in a `.npz` archive alongside a metadata JSON file.

```
saves/
└── save_001/
    ├── meta.json        # tick count, grid size, config snapshot
    └── state.npz        # all arrays compressed
```

```python
# Save
np.savez_compressed("saves/save_001/state.npz", 
    altitude=world.altitude,
    ground_water=world.front.ground_water,
    # ...
)

# Load
data = np.load("saves/save_001/state.npz")
world.altitude = data["altitude"]
# ...
```

---

## Configuration

At startup, `config/default.json` is loaded into a plain Python dict and passed to all phase modules. Parameters are never hardcoded — always read from config.

```python
import json

def load_config(path="config/default.json") -> dict:
    with open(path) as f:
        return json.load(f)
```

---

## Build milestones

| # | Milestone | Deliverable |
|---|-----------|-------------|
| 1 | **Data structures** | `World` class with all numpy arrays, config loading |
| 2 | **World generation** | Simplex noise altitude, initial water/temp distribution |
| 3 | **Tick loop** | All 8 phases implemented, console validation (conservation laws) |
| 4 | **2D map view** | Pygame window, one color per base type, step/auto controls |
| 5 | **Overlays** | Temperature, water, pressure as color gradients |
| 6 | **UI panels** | Text frame, pop-ups, parameter display |
| 7 | **3D torus view** | ModernGL, instanced vegetation models |

---

## Dependencies

```
numpy          # world state arrays and all vectorized operations
pygame         # 2D rendering, event loop, UI
moderngl       # 3D torus view (milestone 7)
noise          # simplex noise for world generation (library: opensimplex or similar)
```

```
pip install numpy pygame moderngl opensimplex
```
