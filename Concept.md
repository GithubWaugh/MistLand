# Concept Notes

## Game concept

Simulation of a finite world, with vegetation, simple living entities, and basic meteorologic behaviors.

---

## Time

### Ticks

One tick = one day.

Player can choose :
- **Automatic** time evolution, with variable speed (0.25 to 50 ticks/second) ;
- **Step by step** — via a "one step forward" button (Space) ;
- **Pause / resume** — via the A key.

### Simulation phases order (per tick)

Each phase is followed by a buffer swap + sync, so every phase reads a fully up-to-date world state.

| # | Phase | Description |
|---|-------|-------------|
| 1 | **Temperature** | Ground ↔ atmosphere net flux ; ground ↔ neighbours radiation |
| 2 | **Pressure** | Derived from atmospheric temperature and altitude |
| 3 | **Vegetation** | Growth / devolution, water and nutriments consumption |
| 4 | **Nutriments** | Diffusion to neighbours |
| 5 | **Evaporation** | Ground water → atmosphere (mist) |
| 6 | **Atmospheric movements** | Wind transports mist and temperature |
| 7 | **Water movements** | Runoff based on hydraulic altitude |
| 8 | **Rain** | Mist → ground water, based on temperature and humidity |

---

## Game appearance

Windowed or full-screen. Current implementation : single pygame window with overlays and inspect panel.

- **Top-down view of the map**
  - Base color for the ground layer (bare/sand/soil) ; lake cells shown in blue
  - Overlayed icons for the vegetation (with transparency) ; submerged lichen shown as algae (~)
  - Mist overlay : white veil, opacity proportional to airborne water (toggle 6)
  - Exclusive data overlays (toggle 1–3, 5) :
    - Water : ground water level
    - Temperature : ground temperature gradient
    - Pressure : atmospheric pressure
    - Altitude : spectral color map (violet=low → red=high)
  - Inspect panel (toggle I) : cell info following the mouse cursor

- **Text frame** — command-line type interaction *(planned)*
- **3D view of the torus** — ModernGL, instanced vegetation models *(planned)*

---

## World

Toric planet, modeled as a 2D grid (wrap-around on both axes u and v).

### Global parameters

Settings defined when starting a new game :

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_width` | 512 | Grid width (u axis) |
| `grid_height` | 256 | Grid height (v axis) |
| `total_water` | 0.5 per cell | Total water in the world (conserved) |
| `total_energy` | — | Total energy in the world (controllable) |

**Data** : grids dumped to files when saving game progress.

---

## World cells

Each cell is defined by a set of static properties and layered data.

### Static properties

- **UV coordinates**
  - `vector2 float32`, normalized (u, v) position of each cell on the torus.

- **Altitude**
  - `float32`, generated via seamlessly-tiling value noise (6 octaves) + gaussian blur.
  - Tiling guaranteed : lattice period = frequency for each octave.
  - *(Erosion algorithm planned for a later version)*

- **Fertility**
  - `float32 [-1..1]`, static field generated once at world creation.
  - Positive = locally favourable to life ; negative = hostile.
  - Noise frequency varies by base type : high frequency for bare rock (sharp patchwork), medium for sand, low for soil (broad zones).
  - Modulates vegetation growth/devolution thresholds.

- **Base type** : `bare` | `sand` | `soil`
  - `soil` = sand + sufficient organic nutriments
  - Each type has constant properties (defined in config) :

| Base type | Thermal inertia | Water retention | Flooding threshold |
|-----------|----------------|-----------------|-------------------|
| `bare` | high | none | 0.2 |
| `sand` | low | low | 0.5 |
| `soil` | medium | medium | 0.8 |

---

### Ground layer *(varying)*

| Field | Type | Description |
|-------|------|-------------|
| `ground_water` | `float32 [0..+∞)` | Water at ground level. Above flooding_threshold = lake cell. |
| `nutriments` | `uint8 [0..255]` | Organic nutriment quantity |
| `ground_temp` | `float32` (°C) | Ground-level temperature |
| `albedo` | `float32 [0..1]` | Energy reflection coefficient ; updated each tick |
| `vegetation_water` | `float32` | Water stored inside plants (part of conserved total) |

---

### Atmospheric layer *(varying)*

| Field | Type | Description |
|-------|------|-------------|
| `pressure` | `float32` (bar) | Drives wind — see Atmosphere section |
| `mist` | `float32 [0..7]` | Airborne water quantity (continuous float, no integer rounding) |
| `atmo_temp` | `float32` (°C) | Atmospheric temperature |

---

### Vegetation layer *(varying)*

Type : `None` | `Lichens` | `Grass` | `Shrubs` | `Trees`

Each level implies all lower levels are present (e.g. `Shrubs` implies `Grass` and `Lichens`).

When a cell is flooded (ground_water ≥ flooding_threshold), only `Lichens` can survive (displayed as algae).

| Field | Type | Description |
|-------|------|-------------|
| `vegetation` | `uint8 [0..4]` | Current vegetation level |
| `veg_tick_counter` | `uint16` | Ticks since last vegetation change |

---

### Entities layer *(not implemented yet)*

List of entities present on the cell.

---

### Description *(derived)*

A human-readable label derived from base type, ground water and vegetation.
Examples : *Desert, Swamp, Rainforest, Lake, Tundra…*

---

## Varying data — evolution rules

### Ground level

#### Temperature

Conservation : total energy (ground_temp + atmo_temp summed over all cells) is constant.

Each tick, a **net flux** is computed between ground and atmosphere :
```
net_flux = exchange_rate × (ground_temp − atmo_temp)
```
What leaves the ground enters the atmosphere exactly — no energy created.

Ground also radiates to its 4 neighbours proportionally to thermal inertia (conservative by construction).

#### Water

Fixed world total quantity (conserved). Total = `ground_water.sum() + mist.sum() × MIST_UNIT + vegetation_water.sum()`

- **Evaporation** : if ground temperature > `evap_temp_threshold` and ground water > 0, a fraction transfers to mist (float, no rounding loss)
- **Runoff** : driven by **hydraulic altitude** rather than terrain altitude alone :
  ```
  effective_alt = altitude + ground_water × water_to_altitude
  ```
  Water only flows toward cells where the water *surface* is lower — prevents overflow into already-flooded cells.
  Sand and soil retain a minimum amount (`retention_min`).
- **Flooding** : if `ground_water` ≥ `flooding_threshold` (varies by base type), the cell is a *Lake* :
  - Displayed in blue on the map
  - Generates nutriments for neighbours each tick
  - Runoff from lakes carries nutriments downstream

#### Nutriments

Each tick, a fraction (`nutriment_diffusion_rate`) of a cell's nutriments diffuses to its 4 direct neighbours.
*(Simulates microfauna movement — insects, worms, etc.)*

| Sources | Sinks |
|---------|-------|
| Vegetation devolution | Vegetation growth |
| Lake proximity | — |

#### Vegetation

Vegetation evolves at most once every `vegetation_growth_period` ticks. When a change occurs (growth or devolution), the counter resets to a **random value in [0, growth_period/2]** to desynchronise patches.

**Fertility modulation** : the static `fertility` field adjusts thresholds locally :
- `effective_water_min = water_min − fertility × 0.10`
- `effective_nut_min = nut_min − fertility × 5`
- Cells with `fertility ≥ 0.6` are immune to devolution (stress tolerance)

**Growth conditions** (all must be met) :

| Condition | Effect |
|-----------|--------|
| `ground_water` ≥ `eff_water_min` AND `temp_min` ≤ temp ≤ `temp_max` AND `nutriments` ≥ `eff_nut_min` AND timer ready | Evolves upward (Lichens → Grass → Shrubs → Trees) |
| Water or temp out of range AND not immune | Devolves downward |

**Flooding rules** (applied every tick, immediate) :
- Flooded cells : `veg > Lichens` → forced devolution by one step per tick toward Lichens
- Lichen on flooded cells survives and is displayed as algae (~) in the UI

**Bank rules** (cells adjacent to a flooded cell) :
- Adjacent to lake, `base_type == bare` → maximum vegetation = Grass
- Adjacent to lake, `base_type == sand` → maximum vegetation = Shrubs

Additional rules :
- Growth consumes `growth_cost_water` (moved to `vegetation_water`) and `growth_cost_nutriments`
- **Lichens** : grow from nothing (no nutriments required) ; release extra nutriments when destroyed
- Devolution returns all `vegetation_water` to ground + releases nutriments

---

### Atmospheric level

#### Pressure

Computed each tick :
```
pressure(cell) = P_base
               − k_temp × (atmo_temp(cell) − temp_ref)
               + k_alt  × altitude(cell)
```
- `temp_ref` : reference temperature at which pressure = P_base
- Hot air (above temp_ref) → low pressure ; cold air → high pressure
- High altitude → low pressure

A `pressure_damping` coefficient smooths convergence and prevents numerical oscillation.

**Latitudinal gradient** : initial temperatures follow `cos(π·v)`, creating a persistent warm/cold banding that drives baseline wind circulation. This gradient is maintained by the temperature phase each tick.

#### Wind

Derived from pressure gradient — not stored, computed each tick :
```
wind(A → B) = k_wind × (pressure(A) − pressure(B))
```
Wind transports mist and atmospheric temperature to neighbours via antisymmetric net flux (exact conservation).

*(Vortices, Coriolis effect — deferred)*

#### Rain

If `atmo_temp` < `rain_temp_threshold` AND `mist` ≥ `rain_humidity_threshold` :
- A fraction (`rain_rate`) of mist transfers to ground water.

Excess mist above 7.0 (from atmospheric transport) precipitates immediately to ground water.

---

### Entities *(not implemented yet)*

- Animals feeding on vegetation and/or each other
- Animals can move to an adjacent cell each tick

---

## Configuration reference

All tunable parameters are stored in `config/default.json`. Keys prefixed `_comment` are documentation only and ignored by the simulation. See `default.json` for per-parameter descriptions.

---

## Future / deferred features

- **O₂ / CO₂** : atmospheric gas cycle (photosynthesis, respiration) — when entities are implemented
- **Energy exchange with outside** : solar gain, radiative loss, albedo-driven
- **Erosion** : altitude smoothing over time based on water flow
- **Large-scale atmospheric dynamics** : vortices, Coriolis effect
- **Geological events** : volcanoes, particle emissions
- **3D torus view** : ModernGL, instanced vegetation models
- **Save / load** : npz archive + metadata JSON
- **Entities** : animals, feeding, movement