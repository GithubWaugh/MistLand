"""
app.py
Pygame renderer for MistLand — refactored overlay system.

Layer stack (bottom → top) :
  Base  : hillshading permanent (luminosité depuis gradient d'altitude)
  L1    : donnée exclusive — soil / temp / pressure / ground-water / air-mist
          Compositing HSL : H+S de L1, L du hillshading
  L2    : éléments sol exclusifs — végétation / nutriments / sédiments (points)
  Mist  : voile atmosphérique (toggle M)
  L3a   : streamers vent + flèche de sens (toggle 3)
  L3b   : pluie (toggle 4)
  Inspect panel

Contrôles :
  [1] cycle L1   [2] cycle L2   [3] cycle L3   [4] wind [5] rain   [M] mist
  [I] inspect    [A] pause      [SPC] step
  [PgUp/Dn] speed    [scroll] zoom    [RMB/MMB] pan
"""

import math
import pygame
import numpy as np
from dataclasses import dataclass

from sim.world import World, VEG_NONE, VEG_LICHENS, VEG_GRASS, VEG_SHRUBS, VEG_TREES
from sim.world import BASE_BARE, BASE_SAND, BASE_SOIL


# ---------------------------------------------------------------------------
# Constants — couleurs
# ---------------------------------------------------------------------------
COLOR_BG        = ( 20,  20,  20)
COLOR_BARE      = ( 90,  85,  80)
COLOR_SAND      = (189, 167, 117)
COLOR_SOIL      = (120,  85,  50)
COLOR_LAKE      = ( 40, 100, 200)
COLOR_RAIN      = (180, 210, 255)
COLOR_SNOW      = (255, 255, 255)

# L2 — points végétation
COLOR_L2_LICHEN = (180, 200, 100)   # jaune-vert pâle
COLOR_L2_GRASS  = (100, 180,  70)   # vert
COLOR_L2_SHRUB  = ( 60, 140,  60)   # vert soutenu
COLOR_L2_TREE   = ( 30,  90,  40)   # vert foncé
# L2 — nutriments
COLOR_L2_NUTR   = (160, 120,  50)   # ocre

MIST_ALPHA_MAX    = 200
TEMP_ALPHA_MAX    = 180
PRES_ALPHA_MAX    = 180
WINDOW_W          = 1024
WINDOW_H          = 512
INFO_BAR_H        = 24
ZOOM_MIN          = 1
ZOOM_MAX          = 32
ZOOM_DEFAULT      = 2
INFO_FONT_SIZE    = 14
INSPECT_FONT_SIZE = 13

SPEED_STEPS   = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
SPEED_DEFAULT = 6

# ---------------------------------------------------------------------------
# Layer 1 — données exclusives
# ---------------------------------------------------------------------------
L1_NONE         = 0
L1_SOIL         = 1
L1_MODES = [L1_NONE, L1_SOIL]
L1_NAMES = {
    L1_NONE        : "---",
    L1_SOIL        : "soil",
}

# ---------------------------------------------------------------------------
# Layer 2 — éléments sol exclusifs
# ---------------------------------------------------------------------------
L2_NONE      = 0
L2_VEG       = 1
L2_NUTRIENTS = 2
L2_SEDIMENTS = 3
L2_MODES = [L2_NONE, L2_VEG, L2_NUTRIENTS, L2_SEDIMENTS]
L2_NAMES = {
    L2_NONE     : "---",
    L2_VEG      : "veg",
    L2_NUTRIENTS: "nutrients",
    L2_SEDIMENTS: "sediments",
}

VEG_NAMES  = {VEG_NONE: "None", VEG_LICHENS: "Lichens",
               VEG_GRASS: "Grass", VEG_SHRUBS: "Shrubs", VEG_TREES: "Trees"}
BASE_NAMES = {BASE_BARE: "Bare", BASE_SAND: "Sand", BASE_SOIL: "Soil"}

# ---------------------------------------------------------------------------
# Layer 3 — éléments atmosphère additifs
# ---------------------------------------------------------------------------
L3_NONE         = 0
L3_TEMP         = 1
L3_PRES         = 2
L3_GRDW         = 3
L3_MODES = [L3_NONE, L3_TEMP, L3_PRES, L3_GRDW]
L3_NAMES = {
    L3_NONE    : "---",
    L3_TEMP    : "temperature",
    L3_PRES    : "pressure",
    L3_GRDW    : "ground-water",
}

# ---------------------------------------------------------------------------
# Layer 4 — Additif : Vent et Pluie (toggle indépendants)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Renderer state
# ---------------------------------------------------------------------------
@dataclass
class RendererState:
    zoom        : int   = ZOOM_DEFAULT
    cam_x       : float = 0.0
    cam_y       : float = 0.0
    layer1_mode : int   = L1_SOIL
    layer2_mode : int   = L2_VEG
    layer3_mode : int   = L3_GRDW
    show_wind   : bool  = False
    show_rain   : bool  = False
    show_mist   : bool  = True
    show_inspect: bool  = False
    paused      : bool  = False
    speed_idx   : int   = SPEED_DEFAULT
    panning     : bool  = False
    pan_start_mx: int   = 0
    pan_start_my: int   = 0
    pan_start_cx: float = 0.0
    pan_start_cy: float = 0.0
    rec_paused  : bool  = False


# ---------------------------------------------------------------------------
# Hillshading
# ---------------------------------------------------------------------------

def _build_hillshade(altitude, snow: np.ndarray) -> np.ndarray:
    dzdx = (np.roll(altitude, -1, axis=1) - np.roll(altitude,  1, axis=1)) * 0.5
    dzdy = (np.roll(altitude, -1, axis=0) - np.roll(altitude,  1, axis=0)) * 0.5

    norm_len = np.sqrt(dzdx ** 2 + dzdy ** 2 + 1.0)
    nx = -dzdx / norm_len
    ny = -dzdy / norm_len
    nz =  1.0  / norm_len

    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    shade = nx * (-inv_sqrt3) + ny * (-inv_sqrt3) + nz * inv_sqrt3

    # Snow effect : brighten areas with snow to make them visually distinct and more visible, especially when the base color is light (e.g., sand or bare)
    # shade = np.where(snow >= 0.01, shade + 0.25, shade)
    shade += snow * 0.25  # Additive brighten based on snow amount, more snow → brighter, without a hard threshold

    # Normalisation : étire les valeurs réelles pour couvrir toute la plage
    s_min, s_max = shade.min(), shade.max()
    if s_max - s_min > 1e-6:
        shade = (shade - s_min) / (s_max - s_min)
    else:
        shade = np.full_like(shade, 0.5)

    # Plage de luminosité cible
    LUM_MIN = 0.5
    LUM_MAX = 0.75
    shade = shade * (LUM_MAX - LUM_MIN) + LUM_MIN 
    
    # Modulation par la neige : éclaircit les zones enneigées
    shade = np.where(snow >= 0.01, shade + 0.25, shade)

    return shade.astype(np.float32)


def _hillshade_to_rgb(shade: np.ndarray) -> np.ndarray:
    """Luminosité scalaire (H, W) → RGB niveaux de gris (H, W, 3) uint8."""
    gray = (shade * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


# ---------------------------------------------------------------------------
# Compositing HSL
# ---------------------------------------------------------------------------

def _rgb_to_hsl(rgb_f: np.ndarray):
    """rgb_f : (H,W,3) float [0,1] → (h, s, l) chacun (H,W) float [0,1]"""
    r, g, b  = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    cmax     = np.maximum(np.maximum(r, g), b)
    cmin     = np.minimum(np.minimum(r, g), b)
    delta    = cmax - cmin
    eps      = 1e-9

    l = (cmax + cmin) * 0.5
    s = np.where(delta < 1e-6, 0.0,
                 delta / (1.0 - np.abs(2 * l - 1) + eps))

    h = np.where(
        delta < 1e-6, 0.0,
        np.where(cmax == r, ((g - b) / (delta + eps)) % 6,
        np.where(cmax == g,  (b - r) / (delta + eps) + 2,
                              (r - g) / (delta + eps) + 4))
    ) / 6.0 % 1.0

    return h, np.clip(s, 0.0, 1.0), np.clip(l, 0.0, 1.0)


def _hsl_to_rgb(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    """h, s, l : (H,W) float [0,1] → (H,W,3) float [0,1]"""
    c  = (1.0 - np.abs(2 * l - 1)) * s
    h6 = h * 6.0
    x  = c * (1.0 - np.abs(h6 % 2 - 1))
    m  = l - c * 0.5

    r = g = b = np.zeros_like(h)
    for i, (rc, gc, bc) in enumerate(
            [(c,x,0),(x,c,0),(0,c,x),(0,x,c),(x,0,c),(c,0,x)]):
        mask = (h6 >= i) & (h6 < i + 1)
        r = np.where(mask, rc, r)
        g = np.where(mask, gc, g)
        b = np.where(mask, bc, b)

    return np.stack([r + m, g + m, b + m], axis=-1)


def _composite_hsl(shade_L: np.ndarray, data_rgb: np.ndarray) -> np.ndarray:
    """
    Compositing HSL : H+S depuis data_rgb, L depuis shade_L.
    shade_L  : (H,W) float [0.1..0.9] luminosité du hillshade
    data_rgb : (H,W,3) uint8
    Retourne (H,W,3) uint8.
    """
    data_f   = data_rgb.astype(np.float32) / 255.0
    h, s, _  = _rgb_to_hsl(data_f)
    result_f = _hsl_to_rgb(h, s, shade_L)
    return (np.clip(result_f, 0.0, 1.0) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Builders RGB pour L1
# ---------------------------------------------------------------------------

def _normalise(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr)
    return ((arr - mn) / (mx - mn)).astype(np.float32)

def _lerp_colour(c0, c1, t):
    t3 = t[:, :, np.newaxis]
    return (np.array(c0) * (1 - t3) + np.array(c1) * t3).astype(np.uint8)

def _compute_flooded(world) -> np.ndarray:
    base_cfg = world.config["base_types"]; bt = world.base_type
    t = np.empty((world.height, world.width), dtype=np.float32)
    t[bt == BASE_BARE] = base_cfg["bare"]["flooding_threshold"]
    t[bt == BASE_SAND] = base_cfg["sand"]["flooding_threshold"]
    t[bt == BASE_SOIL] = base_cfg["soil"]["flooding_threshold"]
    return world.front.ground_water >= t

def _l1_soil_rgb(world) -> np.ndarray:
    h, w  = world.height, world.width
    rgb   = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[world.base_type == BASE_BARE] = COLOR_BARE
    rgb[world.base_type == BASE_SAND] = COLOR_SAND
    rgb[world.base_type == BASE_SOIL] = COLOR_SOIL
    bt       = world.base_type
    base_cfg = world.config["base_types"]
    ft       = np.empty((h, w), dtype=np.float32)
    ft[bt == BASE_BARE] = base_cfg["bare"]["flooding_threshold"]
    ft[bt == BASE_SAND] = base_cfg["sand"]["flooding_threshold"]
    ft[bt == BASE_SOIL] = base_cfg["soil"]["flooding_threshold"]
    rgb[world.front.ground_water >= ft] = COLOR_LAKE
    rgb[world.front.ground_snow >= 0.01] = COLOR_SNOW
    return rgb

_L1_BUILDERS = {    
    L1_NONE: lambda w: np.zeros((w.height, w.width, 3), dtype=np.uint8),
    L1_SOIL: _l1_soil_rgb, }



# ---------------------------------------------------------------------------
# Voile de brume (toggle indépendant)
# ---------------------------------------------------------------------------

def _build_mist_surface(world, vw, vh, cx, cy, zoom):
    ww, wh = world.width, world.height
    x0 = int(cx); y0 = int(cy)
    x1 = min(x0 + (-(-vw // zoom) + 1), ww)
    y1 = min(y0 + (-(-vh // zoom) + 1), wh)
    crop = world.front.mist[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    alpha = (crop.astype(np.float32) / 7.0 * MIST_ALPHA_MAX).astype(np.uint8)
    hc, wc = crop.shape
    surf = pygame.Surface((wc, hc), pygame.SRCALPHA)
    surf.fill((255, 255, 255, 0))
    a = pygame.surfarray.pixels_alpha(surf)
    a[:, :] = alpha.T
    del a
    return (pygame.transform.scale(surf, ((x1 - x0) * zoom, (y1 - y0) * zoom)),
            -int((cx - x0) * zoom), -int((cy - y0) * zoom))


# ---------------------------------------------------------------------------
# L2 — Végétation (points, zoom-adaptive)
# ---------------------------------------------------------------------------

# (couleur RGB, alpha [0-255], densité relative)
_VEG_POINT_STYLE = {
    VEG_LICHENS: (COLOR_L2_LICHEN, 128, 0.15),   # 0.15 épars, semi-transparent
    VEG_GRASS  : (COLOR_L2_GRASS,  128, 0.30),   # 0.30 plus dense, semi-transparent
    VEG_SHRUBS : (COLOR_L2_SHRUB,  200, 0.55),   # 0.55 encore plus dense, plus opaque
    VEG_TREES  : (COLOR_L2_TREE,   255, 0.80),   # dense, opaque
}


def _draw_l2_veg(surface, world, cam_x, cam_y, zoom, vw, vh, is_flooded):
    veg = world.front.vegetation
    x0  = int(cam_x); y0 = int(cam_y)
    x1  = min(x0 + (-(-vw // zoom) + 1), world.width)
    y1  = min(y0 + (-(-vh // zoom) + 1), world.height)
    h_crop = y1 - y0; w_crop = x1 - x0
    if h_crop <= 0 or w_crop <= 0:
        return
    
    color_variation = 25  # variation de couleur aléatoire pour éviter l'effet de motif répétitif

    if zoom < 4:
        # Faible zoom : teinte chaque cellule avec la couleur de végétation
        crop_veg = veg[y0:y1, x0:x1]      # (h_crop, w_crop)
        crop_veg = veg[y0:y1, x0:x1].copy()
        crop_flooded = is_flooded[y0:y1, x0:x1]
        crop_veg[crop_flooded] = VEG_NONE   # masque avant de construire la surface
        surf = pygame.Surface((w_crop, h_crop), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        px3d = pygame.surfarray.pixels3d(surf)   # (w_crop, h_crop, 3) — axe x en premier
        pa   = pygame.surfarray.pixels_alpha(surf)
        rng_var = np.random.default_rng(int(y0 * world.width + x0))
        var_wh  = rng_var.integers(-color_variation, color_variation + 1, size=(w_crop, h_crop, 3), dtype=np.int16)
        for lv, (col, alpha, _) in _VEG_POINT_STYLE.items():
            mask   = (crop_veg == lv).T
            varied = np.clip(np.array(col, dtype=np.int16) + var_wh, 0, 255).astype(np.uint8)
            px3d[mask] = varied[mask]
            pa[mask]   = alpha
        del px3d, pa
        scaled = pygame.transform.scale(surf, ((x1 - x0) * zoom, (y1 - y0) * zoom))
        surface.blit(scaled, (-int((cam_x - x0) * zoom), -int((cam_y - y0) * zoom)))

    else:
        # Zoom élevé : points distribués dans chaque cellule
        for gy in range(y0, y1):
            for gx in range(x0, x1):
                cy_w = gy % world.height
                cx_w = gx % world.width
                lv   = int(veg[cy_w, cx_w])
                if lv == VEG_NONE or is_flooded[cy_w, cx_w]:
                    continue
                col, _, density = _VEG_POINT_STYLE[lv]
                n_dots = max(1, int(density * zoom * zoom * 0.5))
                scx = int((gx - cam_x) * zoom)
                scy = int((gy - cam_y) * zoom)
                cell_seed = cy_w * world.width + cx_w
                rng_col   = np.random.default_rng(cell_seed + 999983)
                variation = rng_col.integers(-color_variation, color_variation + 1, size=3, dtype=np.int16)
                varied_col = tuple(np.clip(np.array(col, dtype=np.int16) + variation, 0, 255).tolist())
                rng = np.random.default_rng(cell_seed)
                for _ in range(n_dots):
                    px = scx + int(rng.integers(0, zoom))
                    py = scy + int(rng.integers(0, zoom))
                    pygame.draw.circle(surface, varied_col, (px, py), 1)


# ---------------------------------------------------------------------------
# L2 — Nutriments (points ocres)
# ---------------------------------------------------------------------------

def _draw_l2_nutrients(surface, world, cam_x, cam_y, zoom, vw, vh, is_flooded):
    nutr = world.front.nutriments
    x0   = int(cam_x); y0 = int(cam_y)
    x1   = min(x0 + (-(-vw // zoom) + 1), world.width)
    y1   = min(y0 + (-(-vh // zoom) + 1), world.height)
    h_crop = y1 - y0; w_crop = x1 - x0
    if h_crop <= 0 or w_crop <= 0:
        return

    if zoom < 4:
        crop_nutr = nutr[y0:y1, x0:x1].astype(np.float32) / 255.0
        crop_nutr[is_flooded[y0:y1, x0:x1]] = 0.0   # ← masque
        surf = pygame.Surface((w_crop, h_crop), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        px3d = pygame.surfarray.pixels3d(surf)
        pa   = pygame.surfarray.pixels_alpha(surf)
        # Remplir couleur partout, moduler alpha par quantité
        px3d[:] = np.array(COLOR_L2_NUTR, dtype=np.uint8)
        pa[:, :] = (crop_nutr.T * 200).astype(np.uint8)
        del px3d, pa
        scaled = pygame.transform.scale(surf, ((x1 - x0) * zoom, (y1 - y0) * zoom))
        surface.blit(scaled, (-int((cam_x - x0) * zoom), -int((cam_y - y0) * zoom)))

    else:
        for gy in range(y0, y1):
            for gx in range(x0, x1):
                if is_flooded[gy, gx]:
                    continue
                cy_w   = gy % world.height
                cx_w   = gx % world.width
                amount = int(nutr[cy_w, cx_w])
                if amount == 0:
                    continue
                density = amount / 255.0
                n_dots  = max(1, int(density * zoom * zoom * 0.4))
                scx = int((gx - cam_x) * zoom)
                scy = int((gy - cam_y) * zoom)
                rng = np.random.default_rng(cy_w * world.width + cx_w)
                for _ in range(n_dots):
                    px = scx + int(rng.integers(0, zoom))
                    py = scy + int(rng.integers(0, zoom))
                    pygame.draw.circle(surface, COLOR_L2_NUTR, (px, py), 1)

# ---------------------------------------------------------------------------
# L3 — Données atmosphère additifs (teinte RGB)
# ---------------------------------------------------------------------------

def _l3_temp_rgb(world) -> np.ndarray:
    return _lerp_colour((50, 80, 200), (220, 60, 30), _normalise(world.front.ground_temp))

def _l3_pressure_rgb(world) -> np.ndarray:
    return _lerp_colour((80, 20, 120), (240, 200, 30), _normalise(world.front.pressure))

def _l3_ground_water_rgb(world) -> np.ndarray:
    flooded = _compute_flooded(world)
    gw = world.front.ground_water * ~flooded
    return _lerp_colour((0, 0, 255), (80, 80, 255),
                        np.clip(gw, 0.0, 1.0).astype(np.float32))

def _draw_l3_temp(surface, world, cam_x, cam_y, zoom, vw, vh):
    rgb = _l3_temp_rgb(world)
    l3_surf, ox, oy = _crop_and_scale(rgb, cam_x, cam_y, zoom, vw, vh, world.width, world.height)
    if l3_surf:
        l3_surf.set_alpha(TEMP_ALPHA_MAX)
        surface.blit(l3_surf, (ox, oy))

def _draw_l3_pressure(surface, world, cam_x, cam_y, zoom, vw, vh):
    rgb = _l3_pressure_rgb(world)
    l3_surf, ox, oy = _crop_and_scale(rgb, cam_x, cam_y, zoom, vw, vh, world.width, world.height)
    if l3_surf:
        l3_surf.set_alpha(PRES_ALPHA_MAX)
        surface.blit(l3_surf, (ox, oy))


_L3_BUILDERS = {
    L3_TEMP    : _l3_temp_rgb,
    L3_PRES    : _l3_pressure_rgb,
    L3_GRDW    : _l3_ground_water_rgb,
}

# ---------------------------------------------------------------------------
# L4a — Streamers de vent + flèche de sens
# ---------------------------------------------------------------------------

def _spectral(value: float) -> tuple:
    """Valeur [0,1] → couleur spectrale (violet→rouge)."""
    value = max(0.0, min(1.0, value))
    stops = [(0.0,(148,0,211)),(0.2,(0,0,255)),(0.4,(0,255,255)),
             (0.6,(0,255,0)),(0.8,(255,255,0)),(1.0,(255,0,0))]
    t0, c0, t1, c1 = 0.0, (0,0,0), 1.0, (255,255,255)
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]; t1, c1 = stops[i + 1]
        if value <= t1:
            break
    t = (value - t0) / (t1 - t0)
    return tuple(round(c0[ch] * (1 - t) + c1[ch] * t) for ch in range(3))


def _draw_wind_streamers(surface, world, cam_x, cam_y, zoom, vw, vh):
    """ Draw vortices as blue circles, then wind streamers with color based on speed and orientation based on direction."""
    for vtx in world.vortex:
        vx, vy, _ = vtx.world_pos(world)
        scx = int((vx - cam_x + 0.5) * zoom)
        scy = int((vy - cam_y + 0.5) * zoom)
        pygame.draw.circle(surface, (0, 0, 139), (scx, scy), 3)
    """Streamers orientés + pointe de flèche indiquant le sens."""
    if zoom < 3:
        return
    x0 = int(cam_x); y0 = int(cam_y)
    x1 = min(x0 + (-(-vw // zoom) + 1), world.width)
    y1 = min(y0 + (-(-vh // zoom) + 1), world.height)
    half      = max(1, zoom * 2 // 5)
    arrow_len = max(1, zoom // 3)
    # Normalisation sur le max réel de la zone visible
    wx = world.front.wind_x
    wy = world.front.wind_y
    speeds = np.sqrt(wx[y0:y1, x0:x1]**2 + wy[y0:y1, x0:x1]**2)
    max_spd = float(speeds.max())
    if max_spd < 1e-6:
        return   # pas de vent du tout → rien à dessiner
    
    wx = world.front.wind_x
    wy = world.front.wind_y

    for gy in range(y0, y1):
        for gx in range(x0, x1):
            cy_w = gy % world.height
            cx_w = gx % world.width
            vx   = float(wx[cy_w, cx_w])
            vy   = float(wy[cy_w, cx_w])
            mag  = math.sqrt(vx * vx + vy * vy)
            if mag < 1e-4:
                continue
            vx /= mag; vy /= mag
            col = _spectral(mag / max_spd)   # toujours [0, 1]


            scx = int((gx - cam_x + 0.5) * zoom)
            scy = int((gy - cam_y + 0.5) * zoom)

            # Corps du streamer
            x_tail = scx - int(vx * half)
            y_tail = scy - int(vy * half)
            x_tip  = scx + int(vx * half)
            y_tip  = scy + int(vy * half)
            pygame.draw.line(surface, col, (x_tail, y_tail), (x_tip, y_tip), 1)

            # Flèche minimaliste à la pointe (deux bras)
            px_p = -vy; py_p = vx    # vecteur perpendiculaire
            ax1 = x_tip - int(vx * arrow_len * 0.7) + int(px_p * arrow_len * 0.4)
            ay1 = y_tip - int(vy * arrow_len * 0.7) + int(py_p * arrow_len * 0.4)
            ax2 = x_tip - int(vx * arrow_len * 0.7) - int(px_p * arrow_len * 0.4)
            ay2 = y_tip - int(vy * arrow_len * 0.7) - int(py_p * arrow_len * 0.4)
            pygame.draw.line(surface, col, (x_tip, y_tip), (ax1, ay1), 1)
            pygame.draw.line(surface, col, (x_tip, y_tip), (ax2, ay2), 1)



# ---------------------------------------------------------------------------
# L4b — Pluie
# ---------------------------------------------------------------------------

def _draw_rain_overlay(surface, world, cam_x, cam_y, zoom, vw, vh):
    cfg = world.config["rain"]
    tt  = cfg["rain_temp_threshold"]
    ht  = cfg["rain_humidity_threshold"]
    x0  = int(cam_x); y0 = int(cam_y)
    x1  = min(x0 + (-(-vw // zoom) + 1), world.width)
    y1  = min(y0 + (-(-vh // zoom) + 1), world.height)
    at  = world.front.atmo_temp[y0:y1, x0:x1]
    ms  = world.front.mist[y0:y1, x0:x1]
    base_cfg = world.config["base_types"]
    bt  = world.base_type[y0:y1, x0:x1]
    ft  = np.empty_like(at)
    ft[bt == BASE_BARE] = base_cfg["bare"]["flooding_threshold"]
    ft[bt == BASE_SAND] = base_cfg["sand"]["flooding_threshold"]
    ft[bt == BASE_SOIL] = base_cfg["soil"]["flooding_threshold"]
    raining  = (at < tt) & (ms >= ht)
    n        = max(1, zoom // 3)
    rows, cols = np.where(raining)
    for row, col in zip(rows, cols):
        px0 = int((x0 + col - cam_x) * zoom)
        py0 = int((y0 + row - cam_y) * zoom)
        for _ in range(n):
            x = px0 + np.random.randint(0, max(1, zoom))
            y = py0 + np.random.randint(0, max(1, zoom))
            if 0 <= x < surface.get_width() and 0 <= y < surface.get_height():
                surface.set_at((x, y), COLOR_RAIN)

# ---------------------------------------------------------------------------
# Panneau d'inspection
# ---------------------------------------------------------------------------

def _draw_inspect(surface, world, mx, my, cam_x, cam_y, zoom, font, is_flooded):
    sw, sh = surface.get_size()
    cx = int(cam_x + mx / zoom)
    cy = int(cam_y + my / zoom)
    if not (0 <= cx < world.width and 0 <= cy < world.height):
        return
    f   = world.front
    bt  = int(world.base_type[cy, cx])
    fld = bool(is_flooded[cy, cx])
    wxv = float(f.wind_x[cy, cx])
    wyv = float(f.wind_y[cy, cx])
    spd = math.sqrt(wxv ** 2 + wyv ** 2)
    #real_alt = (world.altitude[cy, cx] + f.ground_water[cy, cx] * world.config["water"]["water_to_altitude"])
    real_alt = world.surface_altitude()[cy, cx]
    lines = [
        f"Cell [{cx},{cy}]  {BASE_NAMES.get(bt, '?')}",
        f"Alt  : {real_alt:.3f}  ({world.altitude[cy, cx]:.3f})",
        f"GW   : {f.ground_water[cy,cx]:.3f}{'  Lake' if fld else ''}",
        f"Temp : {f.ground_temp[cy,cx]:.1f}°C / {f.atmo_temp[cy,cx]:.1f}°C atmo",
        f"Mist : {f.mist[cy,cx]:.2f}",
        f"Snow : {f.ground_snow[cy,cx] if f.ground_snow[cy,cx]>0 else 'No'}",
        f"Press: {f.pressure[cy,cx]:.4f} bar",
        f"Wind : ({wxv:+.4f}, {wyv:+.4f})  spd={spd:.4f}",
        f"Veg  : {VEG_NAMES.get(int(f.vegetation[cy,cx]), '?')}",
        f"Nutr : {int(f.nutriments[cy,cx])}",
        f"Fert : {world.fertility[cy,cx]:+.3f}",
    ]
    pad = 6; lh = font.get_linesize()
    pw  = max(font.size(l)[0] for l in lines) + pad * 2
    ph  = len(lines) * lh + pad * 2
    px  = mx + 14; py = my + 14
    if px + pw > sw: px = mx - pw - 6
    if py + ph > sh - INFO_BAR_H: py = my - ph - 6
    bg = pygame.Surface((pw, ph), pygame.SRCALPHA)
    bg.fill((10, 10, 10, 200))
    surface.blit(bg, (px, py))
    pygame.draw.rect(surface, (100, 100, 100), (px, py, pw, ph), 1)
    for i, line in enumerate(lines):
        col = (255, 220, 80) if i == 0 else (220, 220, 220)
        surface.blit(font.render(line, True, col), (px + pad, py + pad + i * lh))


# ---------------------------------------------------------------------------
# Helper crop + scale
# ---------------------------------------------------------------------------

def _crop_and_scale(rgb, cam_x, cam_y, zoom, vw, vh, ww, wh):
    x0 = int(cam_x); y0 = int(cam_y)
    x1 = min(x0 + (-(-vw // zoom) + 1), ww)
    y1 = min(y0 + (-(-vh // zoom) + 1), wh)
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return None, 0, 0
    surf   = pygame.surfarray.make_surface(crop.transpose(1, 0, 2))
    scaled = pygame.transform.scale(surf, ((x1 - x0) * zoom, (y1 - y0) * zoom))
    return scaled, -int((cam_x - x0) * zoom), -int((cam_y - y0) * zoom)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class Renderer:
    def __init__(self, world: World):
        self.state        = RendererState()
        self._hillshade_L = _build_hillshade(world.surface_altitude(), world.front.ground_snow) 
        self._font        = None
        self._ifont       = None

    def init_fonts(self):
        self._font  = pygame.font.SysFont("monospace", INFO_FONT_SIZE)
        self._ifont = pygame.font.SysFont("monospace", INSPECT_FONT_SIZE)

    def clamp_camera(self, screen_size, world):
        sw, sh = screen_size; s = self.state
        vw = sw / s.zoom; vh = (sh - INFO_BAR_H) / s.zoom
        s.cam_x = max(0.0, min(s.cam_x, max(0.0, world.width  - vw)))
        s.cam_y = max(0.0, min(s.cam_y, max(0.0, world.height - vh)))

    def render(self, screen: pygame.Surface, world: World):
        if self._font is None:
            self.init_fonts()
        s = self.state
        sw, sh = screen.get_size()
        vw, vh = sw, sh - INFO_BAR_H
        ww, wh = world.width, world.height
        is_flooded = _compute_flooded(world)
        self._hillshade_L = _build_hillshade(world.surface_altitude(),world.front.ground_snow)   # recalculer à chaque frame pour prendre en compte la neige

        # --- Couche de base : hillshade + compositing L1 optionnel ---
        if s.layer1_mode == L1_NONE:
            base_rgb = _hillshade_to_rgb(self._hillshade_L)
        else:
            l1_rgb   = _L1_BUILDERS[s.layer1_mode](world)
            base_rgb = _composite_hsl(self._hillshade_L, l1_rgb)

        base_surf, ox, oy = _crop_and_scale(base_rgb, s.cam_x, s.cam_y,
                                             s.zoom, vw, vh, ww, wh)
        screen.fill(COLOR_BG, (0, 0, sw, vh))
        if base_surf:
            screen.blit(base_surf, (ox, oy))

        # --- L2 : éléments sol ---
        if s.layer2_mode == L2_VEG:
            _draw_l2_veg(screen, world, s.cam_x, s.cam_y,
                         s.zoom, vw, vh, is_flooded)
        elif s.layer2_mode == L2_NUTRIENTS:
            _draw_l2_nutrients(screen, world, s.cam_x, s.cam_y, s.zoom, vw, vh, is_flooded)
        # L2_SEDIMENTS : non encore implémenté

        # --- L3 : données atmosphère additifs ---
        if s.layer3_mode == L3_GRDW:
            x0, y0 = int(s.cam_x), int(s.cam_y)
            x1 = min(x0 + (-(-vw // s.zoom) + 1), ww)
            y1 = min(y0 + (-(-vh // s.zoom) + 1), wh)
            flooded = _compute_flooded(world)
            alpha = np.clip(world.front.ground_water * ~flooded * 255, 0, 255).astype(np.uint8)
            tile = alpha[y0:y1, x0:x1]
            surf = pygame.Surface((tile.shape[1], tile.shape[0]), pygame.SRCALPHA)
            surf.fill((40, 100, 200, 0))
            alpha_arr = pygame.surfarray.pixels_alpha(surf)
            alpha_arr[:, :] = tile.T
            del alpha_arr
            scaled = pygame.transform.scale(surf, ((x1 - x0) * s.zoom, (y1 - y0) * s.zoom))
            ox = -int((s.cam_x - x0) * s.zoom)
            oy = -int((s.cam_y - y0) * s.zoom)
            screen.blit(scaled, (ox, oy))
        elif s.layer3_mode in _L3_BUILDERS:
            l3_rgb = _L3_BUILDERS[s.layer3_mode](world)
            l3_surf, ox, oy = _crop_and_scale(l3_rgb, s.cam_x, s.cam_y,
                                             s.zoom, vw, vh, ww, wh)
            if l3_surf:
                l3_surf.set_alpha(128)   # semi-transparent
                screen.blit(l3_surf, (ox, oy))

        # --- Voile de brume ---
        if s.show_mist:
            r = _build_mist_surface(world, vw, vh, s.cam_x, s.cam_y, s.zoom)
            if r:
                screen.blit(r[0], (r[1], r[2]))

        # --- L3a : vent ---
        if s.show_wind:
            _draw_wind_streamers(screen, world, s.cam_x, s.cam_y, s.zoom, vw, vh)

        # --- L3b : pluie ---
        if s.show_rain and not s.paused:
            _draw_rain_overlay(screen, world, s.cam_x, s.cam_y, s.zoom, vw, vh)

        # --- Inspect ---
        if s.show_inspect and self._ifont:
            mx, my = pygame.mouse.get_pos()
            if my < vh:
                _draw_inspect(screen, world, mx, my,
                              s.cam_x, s.cam_y, s.zoom, self._ifont, is_flooded)

        # --- Barre de statut ---
        pygame.draw.rect(screen, (20, 20, 20), (0, sh - INFO_BAR_H, sw, INFO_BAR_H))
        spd  = f"{SPEED_STEPS[s.speed_idx]:.2f}".rstrip('0').rstrip('.') + " t/s"
        l4st = (("wind" if s.show_wind else "---")
                + "+" + ("rain" if s.show_rain else "---"))
        rec_str = "---" if s.rec_paused else "REC"
        info = (
            f"Tick {world.tick_count:5d}  |  Zoom {s.zoom:2d}x  |  "
            f"[{'PAUSED' if s.paused else spd}]  |  [{rec_str}]  ||  "
            f"L1:{L1_NAMES[s.layer1_mode]}  "
            f"L2:{L2_NAMES[s.layer2_mode]}  "
            f"L3:{L3_NAMES[s.layer3_mode]}  "
            f"L4:{l4st}  |  "
            f"Mist:{'on' if s.show_mist else 'off'}  "
            f"Inspect:{'on' if s.show_inspect else 'off'}  |  "
            f"[1]L1 [2]L2 [3]wind [4]rain [5]pressure [M]mist [I]inspect  "
            f"[A]pause [R]rec [SPC]step [PgUp/Dn]speed"
        )
        screen.blit(self._font.render(info, True, (180, 180, 180)),
                    (8, sh - INFO_BAR_H + 4))
        pygame.display.flip()

    def handle_event(self, event, world, screen_size) -> set:
        actions = set(); s = self.state
        if event.type == pygame.QUIT:
            actions.add('quit')
        elif event.type == pygame.KEYDOWN:
            k = event.key
            if k in (pygame.K_ESCAPE, pygame.K_q):
                actions.add('quit')
            elif k == pygame.K_SPACE:
                if s.paused: actions.add('step')
            elif k == pygame.K_a:
                actions.add('toggle_pause')
            elif k == pygame.K_PAGEUP:
                actions.add('speed_up')
            elif k == pygame.K_PAGEDOWN:
                actions.add('speed_down')
            elif k == pygame.K_1:
                idx = (L1_MODES.index(s.layer1_mode) + 1) % len(L1_MODES)
                s.layer1_mode = L1_MODES[idx]
            elif k == pygame.K_2:
                idx = (L2_MODES.index(s.layer2_mode) + 1) % len(L2_MODES)
                s.layer2_mode = L2_MODES[idx]
            elif k == pygame.K_3:
                idx = (L3_MODES.index(s.layer3_mode) + 1) % len(L3_MODES)
                s.layer3_mode = L3_MODES[idx]
            elif k == pygame.K_4:
                s.show_wind = not s.show_wind
            elif k == pygame.K_5:
                s.show_rain = not s.show_rain
            elif k == pygame.K_m:
                s.show_mist = not s.show_mist
            elif k == pygame.K_i:
                s.show_inspect = not s.show_inspect
            elif k == pygame.K_r:
                actions.add('toggle_rec')
        elif event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            wx = s.cam_x + mx / s.zoom; wy = s.cam_y + my / s.zoom
            s.zoom  = max(ZOOM_MIN, min(ZOOM_MAX, s.zoom + event.y))
            s.cam_x = wx - mx / s.zoom; s.cam_y = wy - my / s.zoom
            self.clamp_camera(screen_size, world)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in (2, 3):
                s.panning = True
                s.pan_start_mx, s.pan_start_my = event.pos
                s.pan_start_cx = s.cam_x; s.pan_start_cy = s.cam_y
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button in (2, 3):
                s.panning = False
        elif event.type == pygame.MOUSEMOTION:
            if s.panning:
                dx = (event.pos[0] - s.pan_start_mx) / s.zoom
                dy = (event.pos[1] - s.pan_start_my) / s.zoom
                s.cam_x = s.pan_start_cx - dx
                s.cam_y = s.pan_start_cy - dy
                self.clamp_camera(screen_size, world)
        elif event.type == pygame.VIDEORESIZE:
            self.clamp_camera(screen_size, world)
        return actions