# Architecture Notes

## Project overview

Python-based world simulation. The backend computes the world state tick by tick in RAM using numpy arrays. The frontend (pygame) reads the current state and renders it. Both run in the same Python process.

---

## Repository structure

```
MistLand/
│
├── config/
│   └── default.json          # All tunable parameters with _comment annotations
│
├── data/
│   └── saves/                # World state dumps (manual save/load — not yet implemented)
│
├── sim/                      # Backend — simulation logic
│   ├── __init__.py
│   ├── world.py              # World class : holds all numpy arrays, exposes tick()
│   ├── generation.py         # World generation : value noise, fertility, initial state
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
│   └── io.py                 # Save / load world state (planned)
│
├── ui/                       # Frontend — rendering and input
│   ├── __init__.py
│   └── app.py                # Main pygame loop, overlays, inspect panel
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
| `altitude` | `float32` | Cell altitude [0..1], generated via seamlessly-tiling value noise (6 octaves) |
| `base_type` | `uint8` | 0 = bare, 1 = sand, 2 = soil |
| `fertility` | `float32` | Local life-favourability [-1..1] ; noise frequency varies by base type |
| `uv` | `float32` | Normalized (u, v) coordinates, shape (height, width, 2) |

#### Ground layer *(updated each tick, stored in WorldBuffers)*

| Array | dtype | Range | Description |
|-------|-------|-------|-------------|
| `ground_water` | `float32` | [0..+∞) | Water at ground level. Above flooding_threshold = lake. |
| `nutriments` | `uint8` | [0..255] | Organic nutriment quantity |
| `ground_temp` | `float32` | °C | Ground temperature |
| `albedo` | `float32` | [0..1] | Energy reflection coefficient |
| `vegetation_water` | `float32` | [0..+∞) | Water stored inside plants (part of conserved total) |

#### Atmospheric layer *(updated each tick)*

| Array | dtype | Range | Description |
|-------|-------|-------|-------------|
| `pressure` | `float32` | bar | Atmospheric pressure (drives wind) |
| `mist` | `float32` | [0..7] | Airborne water — continuous float (no integer rounding artefacts) |
| `atmo_temp` | `float32` | °C | Atmospheric temperature |

#### Vegetation layer *(updated each tick)*

| Array | dtype | Range | Description |
|-------|-------|-------|-------------|
| `vegetation` | `uint8` | [0..4] | 0=None, 1=Lichens, 2=Grass, 3=Shrubs, 4=Trees |
| `veg_tick_counter` | `uint16` | ticks | Ticks since last vegetation change |

---

### Double buffer

The `World` object maintains two `WorldBuffers` instances : **front** and **back**.

- The renderer reads from `front`
- Each phase reads from `front` and writes to `back`
- After each phase : `swap_buffers()` + `sync_back_from_front()` — back becomes the new front for the next phase

```python
# After each phase :
world.front, world.back = world.back, world.front
world.back.copy_from(world.front)   # sync for next phase
```

This pattern ensures every phase sees a fully consistent, up-to-date world state, and inter-phase dependencies are respected (e.g. evaporation updates mist before atmosphere reads it).

---

## Tick loop

```python
def tick(self) -> None:
    self.sync_back_from_front()

    temperature.step(self);  self.swap_buffers(); self.sync_back_from_front()
    pressure.step(self);     self.swap_buffers(); self.sync_back_from_front()
    vegetation.step(self);   self.swap_buffers(); self.sync_back_from_front()
    nutriments.step(self);   self.swap_buffers(); self.sync_back_from_front()
    evaporation.step(self);  self.swap_buffers(); self.sync_back_from_front()
    atmosphere.step(self);   self.swap_buffers(); self.sync_back_from_front()
    water.step(self);        self.swap_buffers(); self.sync_back_from_front()
    rain.step(self);         self.swap_buffers()

    self.tick_count += 1
```

Each phase module exposes a single `step(world)` function that reads from `world.front` and writes to `world.back`.

---

## Conservation laws

Two quantities are strictly conserved across all ticks :

**Total water** = `ground_water.sum() + mist.sum() × MIST_UNIT + vegetation_water.sum()`
- `mist` is `float32` to avoid integer rounding losses
- `water.py` applies a float32 rounding correction after runoff
- `atmosphere.py` converts excess mist > 7.0 to ground water instead of clipping

**Total energy** = `ground_temp.sum() + atmo_temp.sum()`
- `temperature.py` uses a net flux formula : what leaves ground enters atmosphere exactly
- `atmosphere.py` uses antisymmetric net flux for temperature transport

---

## Key implementation notes

### Mist : float32 instead of uint8

`mist` was originally `uint8 [0..7]`. It was changed to `float32 [0..7]` to eliminate integer rounding artefacts (checkerboard pattern in cloud display and water conservation errors).
The `mist_accumulator` field (previously used to buffer fractional evaporation) was removed as a consequence.

### Hydraulic altitude for runoff

Runoff in `water.py` uses hydraulic altitude rather than terrain altitude :
```
effective_alt = altitude + ground_water × water_to_altitude
```
This prevents water from flowing into already-flooded cells that happen to be at lower terrain altitude.

### Seamlessly-tiling noise

`generation.py` uses value noise with `period = frequency` (integer) for each octave. This guarantees `noise(x=0) == noise(x=frequency)` on both axes — perfectly seamless on the torus.

### Vegetation desynchronisation

When a cell grows or devolves, its `veg_tick_counter` resets to a random value in `[0, growth_period/2]` rather than 0. This breaks the synchronous wave behaviour seen in large homogeneous patches.

---

## Threading model

Currently single-threaded : the simulation tick runs in the pygame main loop (accumulator-based timing). No separate thread is needed since tick times are well under 16ms at current grid sizes.

```python
# In pygame loop :
tick_accum += dt
interval = 1.0 / ticks_per_second
while tick_accum >= interval:
    world.tick()
    tick_accum -= interval
```

*(Multi-threading planned if larger grids require it)*

---

## World generation

`generation.py` populates the world in order :

1. **Altitude** : 6-octave value noise (freqs 2, 4, 8, 16, 32, 64) + gaussian blur
2. **Base type** : altitude thresholds (bare < 0.3, sand < 0.6, soil ≥ 0.6)
3. **Fertility** : per-base-type noise at different frequencies, blended into a [-1..1] field
4. **Water distribution** : weighted by inverse altitude (low areas get more water)
5. **Temperature** : latitudinal gradient `cos(π·v)` + altitude lapse rate
6. **Albedo** : from base type

---

## UI — app.py

Single file, all rendering in pygame. Key features :

| Feature | Key | Notes |
|---------|-----|-------|
| Step one tick | Space | Only when paused |
| Pause / resume | A | |
| Speed control | PgUp / PgDn | 8 steps : 0.25 to 50 t/s |
| Water overlay | 1 | Semi-transparent, exclusive |
| Temperature overlay | 2 | Semi-transparent, exclusive |
| Pressure overlay | 3 | Semi-transparent, exclusive |
| Vegetation icons | 4 | Toggle ; hidden below zoom 4x |
| Altitude overlay | 5 | Spectral (violet→red), opaque, exclusive |
| Mist overlay | 6 | White veil, per-pixel alpha |
| Inspect panel | I | Cell info following mouse cursor |
| Zoom | Mouse wheel | 1x – 32x, centred on cursor |
| Pan | RMB / MMB drag | Clamped to world bounds |

Lakes are displayed in blue on the base layer (dynamic, recomputed each frame).
Submerged lichen is displayed as a teal ~ (algae icon).

---

## Save / load *(planned)*

World state will be saved as `.npz` archive + metadata JSON :

```
saves/
└── save_001/
    ├── meta.json        # tick count, grid size, config snapshot
    └── state.npz        # all WorldBuffers arrays compressed
```

---

## Configuration

`config/default.json` is loaded at startup into a plain Python dict. All phase modules read parameters from this dict — nothing is hardcoded. Keys prefixed `_comment` are documentation and are ignored.

---

## Build milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | **Data structures** | ✅ World class, WorldBuffers, config loading |
| 2 | **World generation** | ✅ Value noise, fertility, temperature gradient |
| 3 | **Tick loop + all 8 phases** | ✅ Validated (water + energy conservation) |
| 4 | **2D map view** | ✅ Pygame window, zoom, pan, lake colors |
| 5 | **Overlays + inspect** | ✅ Water, temp, pressure, altitude (spectral), mist, inspect panel |
| 6 | **UI controls** | ✅ Step/auto/speed, all toggle keys |
| 7 | **3D torus view** | ⏳ ModernGL, instanced vegetation models |
| 8 | **Save / load** | ⏳ npz archive + metadata JSON |
| 9 | **Entities** | ⏳ Animals, feeding, movement |

---

## Dependencies

```
numpy          # world state arrays and all vectorized operations
pygame         # 2D rendering, event loop, UI
scipy          # gaussian_filter for noise smoothing
moderngl       # 3D torus view (milestone 7)
```

```
pip install numpy pygame scipy moderngl
```