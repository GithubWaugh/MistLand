"""
app.py
Pygame renderer for MistLand.
Wind streamers now use the persistent wind_x/wind_y field from N-S.
"""

import math
import pygame
import numpy as np
from dataclasses import dataclass

from sim.world import World, VEG_NONE, VEG_LICHENS, VEG_GRASS, VEG_SHRUBS, VEG_TREES
from sim.world import BASE_BARE, BASE_SAND, BASE_SOIL


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLOR_BARE   = (90,  85,  80)
COLOR_SAND   = (189, 167, 117)
COLOR_SOIL   = (120,  85,  50)
COLOR_LAKE   = (40,  100, 200)
COLOR_BG     = (20,  20,  20)
COLOR_LICHEN = (80,  110,  60)
COLOR_ALGAE  = (50,  170, 140)
COLOR_GRASS  = (120, 180,  80)
COLOR_SHRUB  = (60,  130,  60)
COLOR_TREE_C = (30,   90,  40)
COLOR_TREE_T = (80,   55,  30)
COLOR_WIND   = (220, 220, 255)
COLOR_RAIN   = (180, 210, 255)

OVERLAY_ALPHA    = 255
MIST_ALPHA_MAX   = 200
WINDOW_W         = 1024
WINDOW_H         = 512
INFO_BAR_H       = 24
ZOOM_MIN         = 1
ZOOM_MAX         = 32
ZOOM_DEFAULT     = 3
INFO_FONT_SIZE   = 14
INSPECT_FONT_SIZE= 13

SPEED_STEPS  = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
SPEED_DEFAULT = 2

OV_NONE = 0; OV_WATER = 1; OV_TEMP = 2; OV_PRESSURE = 3; OV_ALTITUDE = 5; OV_FERTILITY = 9

VEG_NAMES  = {VEG_NONE:"None", VEG_LICHENS:"Lichens",
              VEG_GRASS:"Grass", VEG_SHRUBS:"Shrubs", VEG_TREES:"Trees"}
BASE_NAMES = {BASE_BARE:"Bare", BASE_SAND:"Sand", BASE_SOIL:"Soil"}


# ---------------------------------------------------------------------------
# Renderer state
# ---------------------------------------------------------------------------
@dataclass
class RendererState:
    zoom        : int   = ZOOM_DEFAULT
    cam_x       : float = 0.0
    cam_y       : float = 0.0
    overlay_mode: int   = OV_NONE
    show_veg    : bool  = True
    show_mist   : bool  = True
    show_wind   : bool  = False
    show_rain   : bool  = False
    show_inspect: bool  = False
    paused      : bool  = False
    speed_idx   : int   = SPEED_DEFAULT
    panning     : bool  = False
    pan_start_mx: int   = 0
    pan_start_my: int   = 0
    pan_start_cx: float = 0.0
    pan_start_cy: float = 0.0


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _lerp_colour(c0, c1, t):
    t3 = t[:, :, np.newaxis]
    return (np.array(c0)*(1-t3) + np.array(c1)*t3).astype(np.uint8)

def _normalise(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6: return np.zeros_like(arr)
    return ((arr - mn) / (mx - mn)).astype(np.float32)

def _spectral_colour(t):
    stops = [(0.0,(148,0,211)),(0.2,(0,0,255)),(0.4,(0,255,255)),
             (0.6,(0,255,0)),(0.8,(255,255,0)),(1.0,(255,0,0))]
    rgb = np.zeros((*t.shape, 3), dtype=np.float32)
    for i in range(len(stops)-1):
        t0,c0 = stops[i]; t1,c1 = stops[i+1]
        mask = (t>=t0)&(t<=t1)
        lt = np.where(mask,(t-t0)/(t1-t0),0.0)
        for ch in range(3):
            rgb[:,:,ch] += mask*(c0[ch]*(1-lt)+c1[ch]*lt)
    return rgb.clip(0,255).astype(np.uint8)

def spectral(value):
    value = max(0.0, min(1.0, value))
    stops = [(0.0,(148,0,211)),(0.2,(0,0,255)),(0.4,(0,255,255)),
             (0.6,(0,255,0)),(0.8,(255,255,0)),(1.0,(255,0,0))]
    t0,t1 = 0,0
    c0,c1 = (0,0,0),(255,255,255)
    for i in range(len(stops)-1):
        t0,c0 = stops[i]
        t1,c1 = stops[i+1]
        if value<=t1 :
            break
    new_value = (value - t0) / (t1 - t0)
    rgb = [0,0,0]
    for ch in range(3):
        rgb[ch] = round(c0[ch]*(1-new_value) + c1[ch]*new_value)
    return tuple(rgb)


# ---------------------------------------------------------------------------
# RGB builders
# ---------------------------------------------------------------------------

def _build_base_rgb(world):
    h,w = world.height, world.width
    rgb = np.zeros((h,w,3), dtype=np.uint8)
    rgb[world.base_type==0] = COLOR_BARE
    rgb[world.base_type==1] = COLOR_SAND
    rgb[world.base_type==2] = COLOR_SOIL
    base_cfg = world.config["base_types"]; bt = world.base_type
    ft = np.empty((h,w), dtype=np.float32)
    ft[bt==BASE_BARE]=base_cfg["bare"]["flooding_threshold"]
    ft[bt==BASE_SAND]=base_cfg["sand"]["flooding_threshold"]
    ft[bt==BASE_SOIL]=base_cfg["soil"]["flooding_threshold"]
    rgb[world.front.ground_water>=ft] = COLOR_LAKE
    return rgb

def _build_altitude_rgb(world):
    return _spectral_colour(world.altitude.astype(np.float32))

def _water_rgb(world):
    return _lerp_colour((0,0,0),(30,100,220),
                        np.clip(world.front.ground_water,0,1).astype(np.float32))

def _temp_rgb(world):
    return _lerp_colour((50,80,200),(220,60,30),_normalise(world.front.ground_temp))

def _fertility_rgb(world):
    return _lerp_colour((30,30,30),(30,200,30),_normalise(world.fertility))

def _pressure_rgb(world):
    return _lerp_colour((80,20,120),(240,200,30),_normalise(world.front.pressure))

def _compute_flooded(world):
    base_cfg=world.config["base_types"]; bt=world.base_type
    t=np.empty((world.height,world.width),dtype=np.float32)
    t[bt==BASE_BARE]=base_cfg["bare"]["flooding_threshold"]
    t[bt==BASE_SAND]=base_cfg["sand"]["flooding_threshold"]
    t[bt==BASE_SOIL]=base_cfg["soil"]["flooding_threshold"]
    return world.front.ground_water>=t

def _build_mist_surface(world, vw, vh, cx, cy, zoom):
    ww,wh=world.width,world.height
    x0=int(cx); y0=int(cy)
    x1=min(x0+(-(-vw//zoom)+1),ww); y1=min(y0+(-(-vh//zoom)+1),wh)
    crop=world.front.mist[y0:y1,x0:x1]
    if crop.size==0: return None
    alpha=(crop.astype(np.float32)/7.0*MIST_ALPHA_MAX).astype(np.uint8)
    hc,wc=crop.shape
    surf=pygame.Surface((wc,hc),pygame.SRCALPHA); surf.fill((255,255,255,0))
    a=pygame.surfarray.pixels_alpha(surf); a[:,:]=alpha.transpose(1,0); del a
    return pygame.transform.scale(surf,((x1-x0)*zoom,(y1-y0)*zoom)),\
           -int((cx-x0)*zoom),-int((cy-y0)*zoom)


# ---------------------------------------------------------------------------
# Wind streamers — use wind_x / wind_y directly
# ---------------------------------------------------------------------------

def _draw_wind_streamers(surface, world, cam_x, cam_y, zoom, vw, vh):
    """Draw oriented dashes from the persistent wind vector field."""
    if zoom < 3:
        return
    x0=int(cam_x); y0=int(cam_y)
    x1=min(x0+(-(-vw//zoom)+1),world.width)
    y1=min(y0+(-(-vh//zoom)+1),world.height)
    half = max(1, zoom*2//5)

    wx = world.front.wind_x
    wy = world.front.wind_y

    for gy in range(y0, y1):
        for gx in range(x0, x1):
            cy = gy % world.height   # wrap torique
            cx = gx % world.width
            vx = float(wx[cy, cx])
            vy = float(wy[cy, cx])
            mag = math.sqrt(vx*vx + vy*vy)
            if mag < 1e-4:
                continue
            vx /= mag;  vy /= mag
            scx = int((gx - cam_x + 0.5) * zoom)
            scy = int((gy - cam_y + 0.5) * zoom)
            str_color = spectral(mag*5)
            pygame.draw.line(surface, str_color,
                             (scx - int(vx*half), scy - int(vy*half)),
                             (scx + int(vx*half), scy + int(vy*half)), 1)


# ---------------------------------------------------------------------------
# Rain overlay
# ---------------------------------------------------------------------------

def _draw_rain_overlay(surface, world, cam_x, cam_y, zoom, vw, vh):
    cfg=world.config["rain"]
    tt=cfg["rain_temp_threshold"]; ht=cfg["rain_humidity_threshold"]
    x0=int(cam_x); y0=int(cam_y)
    x1=min(x0+(-(-vw//zoom)+1),world.width)
    y1=min(y0+(-(-vh//zoom)+1),world.height)
    at=world.front.atmo_temp[y0:y1,x0:x1]
    ms=world.front.mist[y0:y1,x0:x1]
    # Exclude flooded cells
    base_cfg=world.config["base_types"]; bt=world.base_type[y0:y1,x0:x1]
    ft=np.empty_like(at)
    ft[bt==BASE_BARE]=base_cfg["bare"]["flooding_threshold"]
    ft[bt==BASE_SAND]=base_cfg["sand"]["flooding_threshold"]
    ft[bt==BASE_SOIL]=base_cfg["soil"]["flooding_threshold"]
    gw=world.front.ground_water[y0:y1,x0:x1]
    raining=(at<tt)&(ms>=ht)#&(gw<ft)
    n=max(1,zoom//3)
    rows,cols=np.where(raining)
    for row,col in zip(rows,cols):
        px0=int((x0+col-cam_x)*zoom); py0=int((y0+row-cam_y)*zoom)
        for _ in range(n):
            pygame.draw.circle(surface,COLOR_RAIN,
                               (px0+np.random.randint(0,max(1,zoom)),
                                py0+np.random.randint(0,max(1,zoom))),1)


# ---------------------------------------------------------------------------
# Vegetation icon
# ---------------------------------------------------------------------------

def _draw_veg_icon(surface, veg_level, px, py, size, submerged=False):
    if veg_level==VEG_NONE or size<2: return
    cx=px+size//2; cy=py+size//2; s=max(1,size)
    if veg_level==VEG_LICHENS:
        if submerged:
            if size>=4:
                w=max(4,s*2//3); amp=max(1,s//5); steps=max(6,w)
                x0=px+(size-w)//2
                pts=[(x0+i*w//steps,cy+int(amp*math.sin(i*2*math.pi/steps*1.5)))
                     for i in range(steps+1)]
                if len(pts)>=2:
                    pygame.draw.lines(surface,COLOR_ALGAE,False,pts,max(1,size//6))
            else: pygame.draw.rect(surface,COLOR_ALGAE,(px,py,s,s))
        else:
            r=max(1,s//3)
            pygame.draw.ellipse(surface,COLOR_LICHEN,(cx-r,cy-max(1,r//2),r*2,max(1,r)))
    elif veg_level==VEG_GRASS:
        if size>=4:
            h=max(2,s//3)
            for off in (-s//4,0,s//4):
                pygame.draw.line(surface,COLOR_GRASS,(cx+off,cy+h//2),(cx+off,cy-h),1)
        else: pygame.draw.rect(surface,COLOR_GRASS,(px,py,s,s))
    elif veg_level==VEG_SHRUBS:
        pygame.draw.circle(surface,COLOR_SHRUB,(cx,cy),max(1,s//3))
    elif veg_level==VEG_TREES:
        r=max(1,s//3); tw=max(1,s//6); th=max(1,s//4)
        pygame.draw.rect(surface,COLOR_TREE_T,(cx-tw//2,cy,tw,th))
        pygame.draw.polygon(surface,COLOR_TREE_C,[(cx,cy-r),(cx-r,cy),(cx+r,cy)])


# ---------------------------------------------------------------------------
# Inspect panel
# ---------------------------------------------------------------------------

def _draw_inspect(surface, world, mx, my, cam_x, cam_y, zoom, font, is_flooded):
    sw,sh=surface.get_size()
    cx=int(cam_x+mx/zoom); cy=int(cam_y+my/zoom)
    if not (0<=cx<world.width and 0<=cy<world.height): return
    f=world.front; bt=int(world.base_type[cy,cx]); fld=bool(is_flooded[cy,cx])
    wxv=float(f.wind_x[cy,cx]); wyv=float(f.wind_y[cy,cx])
    spd=math.sqrt(wxv**2+wyv**2)
    real_altitude = world.altitude[cy,cx] + world.front.ground_water[cy,cx]*world.config["water"]["water_to_altitude"]
    lines=[
        f"Cell [{cx},{cy}]  {BASE_NAMES.get(bt,'?')}",
        f"Alt  : {real_altitude:.3f}",
        f"GW   : {f.ground_water[cy,cx]:.3f}{'  Lake' if fld else ''}",
        f"Temp : {f.ground_temp[cy,cx]:.1f}°C / {f.atmo_temp[cy,cx]:.1f}°C atmo",
        f"Mist : {f.mist[cy,cx]:.2f}",
        f"Press: {f.pressure[cy,cx]:.4f} bar",
        f"Wind : ({wxv:+.2f}, {wyv:+.2f})  spd={spd:.2f}",
        f"Veg  : {VEG_NAMES.get(int(f.vegetation[cy,cx]),'?')}",
        f"Nutr : {int(f.nutriments[cy,cx])}",
        f"Fert : {world.fertility[cy,cx]:+.3f}",
    ]
    pad=6; lh=font.get_linesize()
    pw=max(font.size(l)[0] for l in lines)+pad*2; ph=len(lines)*lh+pad*2
    px=mx+14; py=my+14
    if px+pw>sw: px=mx-pw-6
    if py+ph>sh-INFO_BAR_H: py=my-ph-6
    bg=pygame.Surface((pw,ph),pygame.SRCALPHA); bg.fill((10,10,10,200))
    surface.blit(bg,(px,py))
    pygame.draw.rect(surface,(100,100,100),(px,py,pw,ph),1)
    for i,line in enumerate(lines):
        surface.blit(font.render(line,True,(255,220,80) if i==0 else (220,220,220)),
                     (px+pad,py+pad+i*lh))


# ---------------------------------------------------------------------------
# Crop + scale
# ---------------------------------------------------------------------------

def _crop_and_scale(rgb, cam_x, cam_y, zoom, vw, vh, ww, wh):
    x0=int(cam_x); y0=int(cam_y)
    x1=min(x0+(-(-vw//zoom)+1),ww); y1=min(y0+(-(-vh//zoom)+1),wh)
    crop=rgb[y0:y1,x0:x1]
    if crop.size==0: return None,0,0
    surf=pygame.surfarray.make_surface(crop.transpose(1,0,2))
    scaled=pygame.transform.scale(surf,((x1-x0)*zoom,(y1-y0)*zoom))
    return scaled,-int((cam_x-x0)*zoom),-int((cam_y-y0)*zoom)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class Renderer:
    def __init__(self, world: World):
        self.state        = RendererState()
        self.altitude_rgb = _build_altitude_rgb(world)
        self._font        = None
        self._ifont       = None

    def init_fonts(self):
        self._font  = pygame.font.SysFont("monospace", INFO_FONT_SIZE)
        self._ifont = pygame.font.SysFont("monospace", INSPECT_FONT_SIZE)

    def clamp_camera(self, screen_size, world):
        sw,sh=screen_size; s=self.state
        vw=sw/s.zoom; vh=(sh-INFO_BAR_H)/s.zoom
        s.cam_x=max(0.0,min(s.cam_x,max(0.0,world.width-vw)))
        s.cam_y=max(0.0,min(s.cam_y,max(0.0,world.height-vh)))

    def render(self, screen: pygame.Surface, world: World):
        if self._font is None: self.init_fonts()
        s=self.state; sw,sh=screen.get_size()
        vw,vh=sw,sh-INFO_BAR_H; ww,wh=world.width,world.height
        is_flooded=_compute_flooded(world)

        base_surf,ox,oy=_crop_and_scale(_build_base_rgb(world),
                                        s.cam_x,s.cam_y,s.zoom,vw,vh,ww,wh)
        screen.fill(COLOR_BG,(0,0,sw,vh))
        if base_surf: screen.blit(base_surf,(ox,oy))

        if s.overlay_mode!=OV_NONE:
            if   s.overlay_mode==OV_WATER:    ov=_water_rgb(world)
            elif s.overlay_mode==OV_TEMP:     ov=_temp_rgb(world)
            elif s.overlay_mode==OV_PRESSURE: ov=_pressure_rgb(world)
            elif s.overlay_mode==OV_FERTILITY: ov=_fertility_rgb(world)
            else:                             ov=self.altitude_rgb
            os2,ox2,oy2=_crop_and_scale(ov,s.cam_x,s.cam_y,s.zoom,vw,vh,ww,wh)
            if os2:
                if s.overlay_mode!=OV_ALTITUDE: os2.set_alpha(OVERLAY_ALPHA)
                screen.blit(os2,(ox2,oy2))

        if s.show_veg and s.zoom>=4:
            x0=int(s.cam_x); y0=int(s.cam_y)
            x1=min(x0+(-(-vw//s.zoom)+1),ww); y1=min(y0+(-(-vh//s.zoom)+1),wh)
            veg=world.front.vegetation
            for cy in range(y0,y1):
                for cx in range(x0,x1):
                    lv=int(veg[cy,cx])
                    if lv==VEG_NONE: continue
                    _draw_veg_icon(screen,lv,
                                   int((cx-s.cam_x)*s.zoom),
                                   int((cy-s.cam_y)*s.zoom),
                                   s.zoom,bool(is_flooded[cy,cx]))

        if s.show_mist:
            r=_build_mist_surface(world,vw,vh,s.cam_x,s.cam_y,s.zoom)
            if r: screen.blit(r[0],(r[1],r[2]))

        if s.show_wind:
            _draw_wind_streamers(screen,world,s.cam_x,s.cam_y,s.zoom,vw,vh)

        if s.show_rain and not s.paused:
            _draw_rain_overlay(screen,world,s.cam_x,s.cam_y,s.zoom,vw,vh)

        if s.show_inspect and self._ifont:
            mx,my=pygame.mouse.get_pos()
            if my<vh:
                _draw_inspect(screen,world,mx,my,
                              s.cam_x,s.cam_y,s.zoom,self._ifont,is_flooded)

        pygame.draw.rect(screen,(20,20,20),(0,sh-INFO_BAR_H,sw,INFO_BAR_H))
        ov_n={OV_NONE:"---",OV_WATER:"water",OV_TEMP:"temp",
              OV_PRESSURE:"pressure",OV_ALTITUDE:"altitude",OV_FERTILITY:"fertility"}
        spd=f"{SPEED_STEPS[s.speed_idx]:.2f}".rstrip('0').rstrip('.')+" t/s"
        info=(f"Tick {world.tick_count:5d}  |  Zoom {s.zoom:2d}x  |  "
              f"[{'PAUSED' if s.paused else spd}]  Ov:{ov_n[s.overlay_mode]}  "
              f"Veg:{'on' if s.show_veg else 'off'}  "
              f"Mist:{'on' if s.show_mist else 'off'}  "
              f"Wind:{'on' if s.show_wind else 'off'}  "
              f"Rain:{'on' if s.show_rain else 'off'}  "
              f"Inspect:{'on' if s.show_inspect else 'off'}  |  "
              f"[SPC]step [A]pause [PgUp/Dn]speed "
              f"[1-3,5,9]overlay [4]veg [6]mist [7]wind [8]rain [I]inspect")
        screen.blit(self._font.render(info,True,(180,180,180)),(8,sh-INFO_BAR_H+4))
        pygame.display.flip()

    def handle_event(self, event, world, screen_size) -> set:
        actions=set(); s=self.state
        if event.type==pygame.QUIT: actions.add('quit')
        elif event.type==pygame.KEYDOWN:
            k=event.key
            if k in (pygame.K_ESCAPE,pygame.K_q): actions.add('quit')
            elif k==pygame.K_SPACE:
                if s.paused: actions.add('step')
            elif k==pygame.K_a:      actions.add('toggle_pause')
            elif k==pygame.K_PAGEUP: actions.add('speed_up')
            elif k==pygame.K_PAGEDOWN: actions.add('speed_down')
            elif k==pygame.K_1: s.overlay_mode=OV_NONE if s.overlay_mode==OV_WATER    else OV_WATER
            elif k==pygame.K_2: s.overlay_mode=OV_NONE if s.overlay_mode==OV_TEMP     else OV_TEMP
            elif k==pygame.K_3: s.overlay_mode=OV_NONE if s.overlay_mode==OV_PRESSURE else OV_PRESSURE
            elif k==pygame.K_4: s.show_veg=not s.show_veg
            elif k==pygame.K_5: s.overlay_mode=OV_NONE if s.overlay_mode==OV_ALTITUDE else OV_ALTITUDE
            elif k==pygame.K_6: s.show_mist=not s.show_mist
            elif k==pygame.K_7: s.show_wind=not s.show_wind
            elif k==pygame.K_8: s.show_rain=not s.show_rain
            elif k==pygame.K_9: s.overlay_mode=OV_NONE if s.overlay_mode==OV_FERTILITY else OV_FERTILITY
            elif k==pygame.K_i: s.show_inspect=not s.show_inspect
        elif event.type==pygame.MOUSEWHEEL:
            mx,my=pygame.mouse.get_pos()
            wx=s.cam_x+mx/s.zoom; wy=s.cam_y+my/s.zoom
            s.zoom=max(ZOOM_MIN,min(ZOOM_MAX,s.zoom+event.y))
            s.cam_x=wx-mx/s.zoom; s.cam_y=wy-my/s.zoom
            self.clamp_camera(screen_size,world)
        elif event.type==pygame.MOUSEBUTTONDOWN:
            if event.button in (2,3):
                s.panning=True
                s.pan_start_mx,s.pan_start_my=event.pos
                s.pan_start_cx=s.cam_x; s.pan_start_cy=s.cam_y
        elif event.type==pygame.MOUSEBUTTONUP:
            if event.button in (2,3): s.panning=False
        elif event.type==pygame.MOUSEMOTION:
            if s.panning:
                dx=(event.pos[0]-s.pan_start_mx)/s.zoom
                dy=(event.pos[1]-s.pan_start_my)/s.zoom
                s.cam_x=s.pan_start_cx-dx; s.cam_y=s.pan_start_cy-dy
                self.clamp_camera(screen_size,world)
        elif event.type==pygame.VIDEORESIZE:
            self.clamp_camera(screen_size,world)
        return actions
