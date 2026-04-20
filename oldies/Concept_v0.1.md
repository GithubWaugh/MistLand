# Concept Notes

## Game concept

Simulation of a finite world, with vegetation, simple living entities, and basic meteorologic behaviors.

---

## Time

### Ticks
One tick = one day.
Player can chose :
- automatic time evolution, by default one tick every second ;
- step by step, by a click on a "one step forward" button.

### Simulation phases order (per tick)

1. **Temperature** — propagation between neighbours (ground + atmosphere)
2. **Pressure** — derived from atmospheric temperature and altitude
3. **Vegetation** — growth / devolution, consumption of water and nutriments
4. **Nutriments** — diffusion to neighbours
5. **Evaporation** — ground water → atmosphere
6. **Atmospheric movements** — wind transports water and particles
7. **Water movements** — runoff from high ground to lower neighbours
8. **Rain** — atmosphere water and temperature → ground

---

## Game appearance

Windowed or full-creen, with several frames :

- **Top-down view of the map**
  - Base color / bitmap for the ground layer
  - Overlayed icons for the vegetation (with transparency)
  - Global overlays to show additional information (atmospheric hygrometry, temperature, wind, pressure, etc.)
- **Text frame** for command-line type interaction with the simulation
- **3D view of the torus**
  - Instancing of low-poly models for vegetation and entities (to be defined later)
- **Pop-up windows**
  - Display of specific informations
  - Parameters settings.

---

## World

Toric planet, modeled as a 2D grid (wrap-around on both axes u and v).

**Global parameters**
Global settings that can be modified when starting a new game :
- Grid Size : default 1024x512
- Finite total water quantity : default 2 units per grid square
- Finite total energy

**Data** : grids, dumped to files when saving game progress.

---

## World cells

Each cell has properties :

- **Altitude** — float number, generated via simplex 2D noise + blur
  - simple erosion algorithm to give more realistic shapes (to be implemented later)
- **Base type** :
  - | `bare` | `sand` | `soil` *(soil = sand + enough organic nutriments)*
  - Each type has some constant properties : 
    - Thermal inertia
    - Water retention capacity
- **Ground layer**
  - Nutriments :
    - int 8bits, varying
  - Ground Water :
    - normalized float, varying
  - Ground temperature :
    - float, varying ;
    - unit : Celsius degrees
  - Albedo :
    - normalized float, varying ;
    - Based on layer type, content of ground layer and vegetation present ;
    - Defines the quantity of energy transfered from the atmosphere to the ground.
  
- **Atmospheric  layer**
  - Atmospheric pressure :
    - float, varying
    - Unit : bar
    *(wind is derived from this — see Atmosphere)*
  - Mist (aka Airborne water) :
    - int, varying, from 0 to 7
  - Atmosphere temperature :
    - float, varying ;
    - unit : Celsius degrees
    

- **Vegetation Layer** 
  - Type :
    | `None` | `Lichens` | `Grass` | `Shrubs` | `Trees` |
  *(Each vegetation level implies previous lower levels is present)*

- **Entities Layer** — list of entities *(to be implemented later)*

- **Description**
  - A generic name derived from Base Type, Quantity of water and Vegetation.
    - String
    - Ex: Swamp, Rainforest, Desert, Lake, etc.

---

## Varying data
The content of every cell's layer will change over time, at each tick.
### At ground level
#### Temperature
The world has a fixed total energy, controlable by the player.  
*(Later version : exchange with outside — solar gain, radiative loss, albedo-dependent)*
- Ground temperature partially radiates :
  - to neighbouring cells, based on cell type thermal inertia
  - to the atmosphere.
- Atmosphere temperature partially radiates to the ground ;

#### Water
Fixed world total quantity (conserved).

- **Evaporation** : if temperature exceeds a threshold and ground water is present, some is transferred to the atmosphere
- **Runoff** : if a cell is higher ground than its neighbours, part of its water flows to them, in proportions depending on altitude delta; sand and soil retain a minimum amount even on high ground
- **Flooding** : 
  - presence of liquid water if quantity above a threshold depending on the base type (ex: Bare, 0.2; Soil : 0.8)
  - if liquid water present :
    - cell is a "Lake" and provides nutriments to its neighbours
    - if water runs to neighbour cells (because of altitude delta), some nutriments travel with it

#### Vegetation

| Condition | Effect |
|-----------|--------|
| Water present + temperature in range + nutriments present | Vegetation evolves upward (e.g. Lichens → Grass → Shrubs → Trees) |
| No water, or temperature out of range | Vegetation devolves downward (e.g. Trees → Shrubs) |

- When vegetation grows, it consumes water and nutriments, which are stored within the plant
- Exception : **Lichens** grow from nothing (no nutriments required) but release nutriments when destroyed
- Vegetation stores some water
- Devolution releases nutriments to the ground and water to the atmosphere
- Vegetation reduces cell's albedo.

---

#### Nutriments

Each tick, a small fraction of a cell's nutriments diffuses to its direct neighbours.  
*(Simulates microfauna movement — insects, worms, etc.)*

Sources :
- Vegetation devolution ;
- Lake proximity.

Sink :
- Vegetation growth (nutriments are consumed and stored)


### At atmospheric level

#### Pressure

Atmospheric pressure is the primary driver of wind. It is computed each tick from two contributions :

```
pressure(cell) = P_base
               - k_temp × atmospheric_temperature(cell)
               + k_alt  × altitude(cell)
```

- **Hot air → low pressure** (air expands and rises)
- **High altitude → low pressure** (less air mass above)

The two effects combine : a hot lowland has very low pressure (both terms reduce it), while a cold highland has ambiguous pressure (effects partially cancel), generating varied wind patterns.

`k_temp` and `k_alt` are tunable parameters (JSON config).  
A damping coefficient should also be defined to prevent numerical oscillation.

#### Wind

Wind is derived from the pressure gradient between neighbouring cells :

```
wind(A → B) = k_wind × (pressure(A) - pressure(B))
```

Wind is not stored — it is computed each tick from the pressure grid.
Each tick :
- The wind transports the cell's atmospheric content (mist and temperature) to its neighbours' atmospheric content, proportionally to wind strength ;
- Temperature partially radiates to the ground ;
- If atmospheric temperature falls below a threshold and airborne water exceeds a humidity threshold, it rains : water transfers from atmosphere to ground

*(Wider scale rules such as vortices and Coriolis effect may be added in a later version)*

---

### Entities
**Not implemented yet.**
- Animals feeding on vegetation and/or each other.
- Animals can move to an adjacent cell.


---

## Future / deferred features

- **O₂ / CO₂** : atmospheric gas cycle (photosynthesis, respiration) — to be added when entities are implemented
- **Energy exchange with outside** : solar gain, radiative loss, albedo
- **Large-scale atmospheric dynamics** : vortices, Coriolis effect
