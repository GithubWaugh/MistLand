"""
app.py
Milestone 4 : pygame window — 2D top-down map view.
Resizable window with zoom, pan, overlays and vegetation icons.

Layers (bottom to top) :
  1. Base map     : base_type colours — always visible
  2. Data overlay : water / temperature / pressure / altitude
                    exclusive, toggle 1/2/3/5
  3. Veg icons    : drawn per cell — toggle 4
  4. Mist overlay : white veil, opacity = mist level — toggle 6

Controls :
  SPACE       : step one tick (when paused)
  A           : pause / resume
  PageUp      : speed up
  PageDown    : slow down
  1           : toggle water overlay
  2           : toggle temperature overlay
  3           : toggle pressure overlay
  4           : toggle vegetation icons
  5           : toggle altitude overlay (static)
  6           : toggle mist / cloud overlay
  Mouse wheel : zoom in/out
  RMB / MMB   : pan
  ESC / Q     : quit
"""

import pygame
import numpy as np
import time

from sim.world import World, VEG_NONE, VEG_LICHENS, VEG_GRASS, VEG_SHRUBS, VEG_TREES


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
COLOR_BARE      = (90,  85,  80)
COLOR_SAND      = (210, 185, 130)
COLOR_SOIL      = (120,  85,  50)
COLOR_BG        = (20,  20,  20)

COLOR_LICHEN    = (80,  110,  60)
COLOR_GRASS     = (120, 180,  80)
COLOR_SHRUB     = (60,  130,  60)
COLOR_TREE_C    = (30,   90,  40)
COLOR_TREE_T    = (80,   55,  30)

OVERLAY_ALPHA   = 160
MIST_ALPHA_MAX  = 200   # max opacity for mist=7

# UI
WINDOW_TITLE    = "MistLand"
WINDOW_W        = 1024
WINDOW_H        = 512
INFO_BAR_H      = 24
ZOOM_MIN        = 1
ZOOM_MAX        = 32
ZOOM_DEFAULT    = 4
INFO_FONT_SIZE  = 14

SPEED_STEPS     = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
SPEED_DEFAULT   = 2

# Overlay modes (exclusive)
OV_NONE     = 0
OV_WATER    = 1
OV_TEMP     = 2
OV_PRESSURE = 3
OV_ALTITUDE = 5


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _lerp_colour(c0, c1, t: np.ndarray) -> np.ndarray:
    t3 = t[:, :, np.newaxis]
    return (np.array(c0) * (1 - t3) + np.array(c1) * t3).astype(np.uint8)


def _normalise(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


# ---------------------------------------------------------------------------
# RGB builders
# ---------------------------------------------------------------------------

def _build_base_rgb(world: World) -> np.ndarray:
    h, w = world.height, world.width
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[world.base_type == 0] = COLOR_BARE
    rgb[world.base_type == 1] = COLOR_SAND
    rgb[world.base_type == 2] = COLOR_SOIL
    return rgb


def _build_altitude_rgb(world: World) -> np.ndarray:
    t   = world.altitude.astype(np.float32)
    rgb = np.zeros((world.height, world.width, 3), dtype=np.uint8)

    mask1 = t < 0.4
    t1    = (t / 0.4).clip(0, 1)
    rgb[mask1] = _lerp_colour((20, 60, 160), (80, 140, 60), t1)[mask1]

    mask2 = (t >= 0.4) & (t < 0.7)
    t2    = ((t - 0.4) / 0.3).clip(0, 1)
    rgb[mask2] = _lerp_colour((80, 140, 60), (130, 100, 60), t2)[mask2]

    mask3 = t >= 0.7
    t3    = ((t - 0.7) / 0.3).clip(0, 1)
    rgb[mask3] = _lerp_colour((130, 100, 60), (240, 240, 240), t3)[mask3]

    return rgb


def _water_rgb(world: World) -> np.ndarray:
    t = np.clip(world.front.ground_water, 0.0, 1.0).astype(np.float32)
    return _lerp_colour((0, 0, 0), (30, 100, 220), t)


def _temp_rgb(world: World) -> np.ndarray:
    t = _normalise(world.front.ground_temp)
    return _lerp_colour((50, 80, 200), (220, 60, 30), t)


def _pressure_rgb(world: World) -> np.ndarray:
    t = _normalise(world.front.pressure)
    return _lerp_colour((80, 20, 120), (240, 200, 30), t)


def _build_mist_surface(world: World, view_w: int, view_h: int,
                        cam_x: float, cam_y: float,
                        zoom: int) -> pygame.Surface | None:
    """
    Build an RGBA surface for the mist overlay.
    Each cell's alpha = (mist / 7) * MIST_ALPHA_MAX.
    White (255, 255, 255) with variable alpha.
    """
    world_w, world_h = world.width, world.height

    cell_x0 = int(cam_x)
    cell_y0 = int(cam_y)
    cells_x  = -(-view_w // zoom) + 1
    cells_y  = -(-view_h // zoom) + 1
    cell_x1  = min(cell_x0 + cells_x, world_w)
    cell_y1  = min(cell_y0 + cells_y, world_h)

    mist_crop = world.front.mist[cell_y0:cell_y1, cell_x0:cell_x1]
    if mist_crop.size == 0:
        return None

    # Alpha channel : proportional to mist level
    alpha = (mist_crop.astype(np.float32) / 7.0 * MIST_ALPHA_MAX).astype(np.uint8)

    # RGBA array : white + alpha
    h_crop, w_crop = mist_crop.shape
    rgba = np.zeros((h_crop, w_crop, 4), dtype=np.uint8)
    rgba[:, :, 0] = 255   # R
    rgba[:, :, 1] = 255   # G
    rgba[:, :, 2] = 255   # B
    rgba[:, :, 3] = alpha

    # Build surface with per-pixel alpha
    # Fill RGB with white, then set alpha channel separately
    surf = pygame.Surface((w_crop, h_crop), pygame.SRCALPHA)
    surf.fill((255, 255, 255, 0))   # white, fully transparent base
    alpha_arr = pygame.surfarray.pixels_alpha(surf)
    alpha_arr[:, :] = alpha.transpose(1, 0)   # (w, h) for pygame
    del alpha_arr   # release lock

    # Scale to zoom
    scaled_w = (cell_x1 - cell_x0) * zoom
    scaled_h = (cell_y1 - cell_y0) * zoom
    scaled   = pygame.transform.scale(surf, (scaled_w, scaled_h))

    return scaled, -int((cam_x - cell_x0) * zoom), -int((cam_y - cell_y0) * zoom)


# ---------------------------------------------------------------------------
# Vegetation icon drawing
# ---------------------------------------------------------------------------

def _draw_veg_icon(surface: pygame.Surface, veg_level: int,
                   px: int, py: int, size: int) -> None:
    if veg_level == VEG_NONE or size < 2:
        return

    cx = px + size // 2
    cy = py + size // 2
    s  = max(1, size)

    if veg_level == VEG_LICHENS:
        r = max(1, s // 3)
        pygame.draw.ellipse(surface, COLOR_LICHEN,
                            (cx - r, cy - max(1, r // 2), r * 2, max(1, r)))
    elif veg_level == VEG_GRASS:
        if size >= 4:
            h = max(2, s // 3)
            for offset in (-s // 4, 0, s // 4):
                pygame.draw.line(surface, COLOR_GRASS,
                                 (cx + offset, cy + h // 2),
                                 (cx + offset, cy - h), 1)
        else:
            pygame.draw.rect(surface, COLOR_GRASS, (px, py, s, s))
    elif veg_level == VEG_SHRUBS:
        r = max(1, s // 3)
        pygame.draw.circle(surface, COLOR_SHRUB, (cx, cy), r)
    elif veg_level == VEG_TREES:
        r       = max(1, s // 3)
        trunk_w = max(1, s // 6)
        trunk_h = max(1, s // 4)
        pygame.draw.rect(surface, COLOR_TREE_T,
                         (cx - trunk_w // 2, cy, trunk_w, trunk_h))
        pts = [(cx, cy - r), (cx - r, cy), (cx + r, cy)]
        pygame.draw.polygon(surface, COLOR_TREE_C, pts)


# ---------------------------------------------------------------------------
# Crop + scale helper
# ---------------------------------------------------------------------------

def _crop_and_scale(rgb: np.ndarray, cam_x: float, cam_y: float,
                    zoom: int, view_w: int, view_h: int,
                    world_w: int, world_h: int) -> tuple:
    cell_x0 = int(cam_x)
    cell_y0 = int(cam_y)
    cells_x  = -(-view_w // zoom) + 1
    cells_y  = -(-view_h // zoom) + 1
    cell_x1  = min(cell_x0 + cells_x, world_w)
    cell_y1  = min(cell_y0 + cells_y, world_h)

    crop = rgb[cell_y0:cell_y1, cell_x0:cell_x1]
    if crop.size == 0:
        return None, 0, 0

    surf     = pygame.surfarray.make_surface(crop.transpose(1, 0, 2))
    scaled_w = (cell_x1 - cell_x0) * zoom
    scaled_h = (cell_y1 - cell_y0) * zoom
    scaled   = pygame.transform.scale(surf, (scaled_w, scaled_h))

    return scaled, -int((cam_x - cell_x0) * zoom), -int((cam_y - cell_y0) * zoom)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(world: World) -> None:
    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_W, WINDOW_H + INFO_BAR_H),
        pygame.RESIZABLE
    )
    pygame.display.set_caption(WINDOW_TITLE)
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("monospace", INFO_FONT_SIZE)

    world_w      = world.width
    world_h      = world.height
    base_rgb     = _build_base_rgb(world)
    altitude_rgb = _build_altitude_rgb(world)

    zoom  = ZOOM_DEFAULT
    cam_x = 0.0
    cam_y = 0.0

    panning      = False
    pan_start_mx = 0
    pan_start_my = 0
    pan_start_cx = 0.0
    pan_start_cy = 0.0

    overlay_mode = OV_NONE
    show_veg     = True
    show_mist    = True   # mist visible by default

    paused     = False
    speed_idx  = SPEED_DEFAULT
    tick_accum = 0.0
    running    = True

    def clamp_camera():
        nonlocal cam_x, cam_y
        sw, sh = screen.get_size()
        view_w = sw / zoom
        view_h = (sh - INFO_BAR_H) / zoom
        cam_x = max(0.0, min(cam_x, max(0.0, world_w - view_w)))
        cam_y = max(0.0, min(cam_y, max(0.0, world_h - view_h)))

    def render():
        sw, sh  = screen.get_size()
        view_w  = sw
        view_h  = sh - INFO_BAR_H

        # Base layer
        base_surf, ox, oy = _crop_and_scale(
            base_rgb, cam_x, cam_y, zoom, view_w, view_h, world_w, world_h)
        screen.fill(COLOR_BG, (0, 0, sw, view_h))
        if base_surf:
            screen.blit(base_surf, (ox, oy))

        # Data overlay (exclusive)
        if overlay_mode != OV_NONE:
            if overlay_mode == OV_WATER:
                ov_rgb = _water_rgb(world)
            elif overlay_mode == OV_TEMP:
                ov_rgb = _temp_rgb(world)
            elif overlay_mode == OV_PRESSURE:
                ov_rgb = _pressure_rgb(world)
            else:
                ov_rgb = altitude_rgb

            ov_surf, ox2, oy2 = _crop_and_scale(
                ov_rgb, cam_x, cam_y, zoom, view_w, view_h, world_w, world_h)
            if ov_surf:
                ov_surf.set_alpha(OVERLAY_ALPHA)
                screen.blit(ov_surf, (ox2, oy2))

        # Vegetation icons
        if show_veg and zoom >= 4:
            cell_x0 = int(cam_x)
            cell_y0 = int(cam_y)
            cells_x = -(-view_w // zoom) + 1
            cells_y = -(-view_h // zoom) + 1
            cell_x1 = min(cell_x0 + cells_x, world_w)
            cell_y1 = min(cell_y0 + cells_y, world_h)
            veg = world.front.vegetation
            for cy in range(cell_y0, cell_y1):
                for cx in range(cell_x0, cell_x1):
                    level = int(veg[cy, cx])
                    if level == VEG_NONE:
                        continue
                    px = int((cx - cam_x) * zoom)
                    py = int((cy - cam_y) * zoom)
                    _draw_veg_icon(screen, level, px, py, zoom)

        # Mist overlay (per-pixel alpha, always on top)
        if show_mist:
            result = _build_mist_surface(
                world, view_w, view_h, cam_x, cam_y, zoom)
            if result is not None:
                mist_surf, mx, my = result
                screen.blit(mist_surf, (mx, my))

        # Info bar
        info_rect = pygame.Rect(0, sh - INFO_BAR_H, sw, INFO_BAR_H)
        pygame.draw.rect(screen, (20, 20, 20), info_rect)

        ov_names = {
            OV_NONE: "---", OV_WATER: "water",
            OV_TEMP: "temp", OV_PRESSURE: "pressure",
            OV_ALTITUDE: "altitude",
        }
        speed_str = f"{SPEED_STEPS[speed_idx]:.2f}".rstrip('0').rstrip('.') + " t/s"
        state_str = "PAUSED" if paused else speed_str

        info_str = (
            f"Tick {world.tick_count:5d}  |  "
            f"Zoom {zoom:2d}x  |  [{state_str}]  "
            f"Overlay:{ov_names[overlay_mode]}  "
            f"Veg:{'on' if show_veg else 'off'}  "
            f"Mist:{'on' if show_mist else 'off'}  |  "
            f"[SPC]step [A]pause [PgUp/Dn]speed "
            f"[1]water [2]temp [3]press [4]veg [5]alt [6]mist  [ESC]quit"
        )
        text_surf = font.render(info_str, True, (180, 180, 180))
        screen.blit(text_surf, (8, sh - INFO_BAR_H + 4))

        pygame.display.flip()

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    if paused:
                        world.tick()
                elif event.key == pygame.K_a:
                    paused = not paused
                    tick_accum = 0.0
                elif event.key == pygame.K_PAGEUP:
                    speed_idx = min(speed_idx + 1, len(SPEED_STEPS) - 1)
                    tick_accum = 0.0
                elif event.key == pygame.K_PAGEDOWN:
                    speed_idx = max(speed_idx - 1, 0)
                    tick_accum = 0.0
                elif event.key == pygame.K_1:
                    overlay_mode = OV_NONE if overlay_mode == OV_WATER    else OV_WATER
                elif event.key == pygame.K_2:
                    overlay_mode = OV_NONE if overlay_mode == OV_TEMP     else OV_TEMP
                elif event.key == pygame.K_3:
                    overlay_mode = OV_NONE if overlay_mode == OV_PRESSURE else OV_PRESSURE
                elif event.key == pygame.K_4:
                    show_veg = not show_veg
                elif event.key == pygame.K_5:
                    overlay_mode = OV_NONE if overlay_mode == OV_ALTITUDE else OV_ALTITUDE
                elif event.key == pygame.K_6:
                    show_mist = not show_mist

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                world_mx = cam_x + mx / zoom
                world_my = cam_y + my / zoom
                zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom + event.y))
                cam_x = world_mx - mx / zoom
                cam_y = world_my - my / zoom
                clamp_camera()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (2, 3):
                    panning      = True
                    pan_start_mx, pan_start_my = event.pos
                    pan_start_cx = cam_x
                    pan_start_cy = cam_y

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (2, 3):
                    panning = False

            elif event.type == pygame.MOUSEMOTION:
                if panning:
                    dx = (event.pos[0] - pan_start_mx) / zoom
                    dy = (event.pos[1] - pan_start_my) / zoom
                    cam_x = pan_start_cx - dx
                    cam_y = pan_start_cy - dy
                    clamp_camera()

            elif event.type == pygame.VIDEORESIZE:
                clamp_camera()

        if not paused:
            tick_accum += dt
            interval = 1.0 / SPEED_STEPS[speed_idx]
            while tick_accum >= interval:
                world.tick()
                tick_accum -= interval

        render()

    pygame.quit()