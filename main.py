###########################################
# Jae-Hoon Sim, KSA of KAIST 2026.04.14
###########################################

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import os
from datetime import datetime
from openpyxl import Workbook
import pygame
import numpy as np

import sys
from pathlib import Path
# -------------------------
# Physical constants + unit mapping (실제 단위 사용)
# -------------------------
KB = 1.380649e-23         # 볼츠만 상수 (J/K)
AVOGADRO_NUMBER = 6.02214076e23        # 아보가드로 수 (1/mol)
MASS = 6.644657e-27 #          # 헬륨 분자 질량 (kg)
MASS_F32 = np.float32(MASS)



ATM_PA = 101325.0         # 표준대기압 (Pa)
KPA_PA = 1000.0           # 1 kPa in Pa

# -------------------------
# SIMULATOR CONST
# -------------------------

AVOGADRO_NUMBER_SIMULATOR = int(2e4) # int(6.02214076 * 1e4)
PARTICLE_NUMBER_SCALE = AVOGADRO_NUMBER / AVOGADRO_NUMBER_SIMULATOR # = 1e19




# 화면길이 스케일: 730 px = 35.51616 cm (730px * 730px * 365px = 0.022414 m^3 = 22.414 L)
# PX_TO_M = 0.3551616 / 730.0  # m/px (1px = 0.3551616 m = 35.51616 cm)
CUBIC_ROOT_OF_VOLUME_PX = 250.0
PX_TO_M = ((22414)**(1/3)/100) / CUBIC_ROOT_OF_VOLUME_PX  # m/px (1px = 0.3551616 m = 35.51616 cm)
PX_TO_M_F32 = np.float32(PX_TO_M)


# 시뮬레이션 시간: 물리 시간 = 시뮬레이션 시간 (재생속도는 PLAYBACK_SCALE로 조절)
T_REF_K = 400.0           # 기준 온도(켈빈). 속도 스케일 기준
_V_REF_M_S = math.sqrt(KB * T_REF_K / (0.5 * MASS))  
#  0.5 * MASS _SIGMA_REF_M_S^2  =  KB * T_REF_K
#  0.5 * kg [_SIGMA_REF_M_S^2] =  energy at reference temperature [J] = [kg m^2/s^2]
# ==> [_SIGMA_REF_M_S] = [meter /second in simulation time]
HIST_VMAX_M_S = 4.0 * _V_REF_M_S
SPEED_SCALE = PX_TO_M  # (m/s) per (px/s)
SPEED_SCALE_F32 = np.float32(SPEED_SCALE)



# Display time scale
PLAYBACK_MIN = 2e-4
PLAYBACK_MAX_PARTICLES = 1.0
PLAYBACK_MAX_MOLS = 1.5
PLAYBACK_SCALE = PLAYBACK_MIN  # time dilation: simulation time per real second
PLAYBACK_LOG_MIN = math.log10(PLAYBACK_MIN)
FAST_MODE_THRESHOLD = 1.0  # fast optimizations apply when scale > this
TIME_AVERAGE = 2e-4  ## For N=1 case, pressure will be averaged over 1e-3 simulation seconds.

RADIUS_MIN = 1.5      # 입자 반지름 최소(픽셀)
RADIUS_MAX = 4.0      # 입자 반지름 최대(픽셀)
# integration stability
MAX_SUBSTEPS =  2  #64  #256
HIST_EVERY = 3  # update histogram every N frames
TABLE_EVERY = 1  # draw table every N frames
# fast playback (mols, scale > 1.0)
FAST_DRAW_EVERY = 2
FAST_HIST_EVERY = 9
FAST_GRAPH_EVERY = 6
FAST_TABLE_EVERY = 3
FAST_DIAG_EVERY = 1  # keep log row rate same as normal (batched dt skips windows)
XLSX_SAVE_EVERY = 20
# -------------------------
# Screen layout (화면 레이아웃 관련 상수)
# -------------------------
SCREEN_W = 1400          # 전체 윈도우의 너비(픽셀)
SCREEN_H = 820           # 전체 윈도우의 높이(픽셀)
TITLE = "Ideal-gas Simulator | 한국과학영재학교 물리지구과학부"  # 윈도우 제목
CREDIT_LINE = "Developed by Department of Physics & Earth Science, KSA of KAIST"

# 시뮬레이션 영역(실린더 위치/크기)
SIM_X0, SIM_X1 = 50, (50+CUBIC_ROOT_OF_VOLUME_PX)     # 실린더(시뮬레이션) 왼쪽/오른쪽 X좌표 width = 730
SIM_Y0, SIM_Y1 = 60, 760     # 실린더(시뮬레이션) 아래/위 Y좌표 height = 700

# 패널(그래프가 그려지는 영역) 위치
PANEL_X0, PANEL_X1 = SIM_X1+50, 1360   # 패널 왼/오른쪽 X좌표
PANEL_Y0, PANEL_Y1 = 60, 790     # 패널 아래/위 Y좌표

# 그래프 패널 레이아웃
PANEL_H = (PANEL_Y1 - PANEL_Y0)           # 패널 높이
GAP = 10                                  # 각 그래프 사이 여백(픽셀)
G_H = (PANEL_H - 3 * GAP) // 4            # 각 그래프 패널 높이
G_W = (PANEL_X1 - PANEL_X0) // 2

# 각 그래프의 영역 정의 (좌, 우, 아래, 위)
G1 = (PANEL_X0, PANEL_X1, PANEL_Y1 - G_H, PANEL_Y1)                           # 속도 분포 히스토그램 패널
G2 = (PANEL_X0, PANEL_X0+G_W-GAP, PANEL_Y1 - 2*G_H - GAP, PANEL_Y1 - G_H - GAP)        # Q(t): 흡수된 열 히스토리 패널
G3 = (PANEL_X0+G_W+GAP, PANEL_X1, PANEL_Y1 - 2*G_H - GAP, PANEL_Y1 - G_H - GAP)  # W(t): 일 히스토리 패널
G4 = (PANEL_X0+G_W+GAP, PANEL_X1, PANEL_Y1 - 3*G_H - 2*GAP, PANEL_Y1 - 2*G_H - 2*GAP)  # P(t): 압력 그래프 패널
G6 = (PANEL_X0, PANEL_X0+G_W-GAP, PANEL_Y0, PANEL_Y1 - 2*G_H - 2*GAP)                 # table panel
PANEL_TITLE_H = 30

# W(t) 아래 공용 영역 (Table | Info | Controls)
BOTTOM_PANEL = (PANEL_X0, PANEL_X1, PANEL_Y0, G3[2] - GAP)
BOTTOM_COL_GAP = 8
BOTTOM_INFO_W = 200
BOTTOM_CONTROLS_W = 260
_bottom_total_w = (BOTTOM_PANEL[1] - BOTTOM_PANEL[0])
_bottom_table_w = _bottom_total_w - (BOTTOM_INFO_W + BOTTOM_CONTROLS_W + 2 * BOTTOM_COL_GAP)
if _bottom_table_w < 280:
    _bottom_table_w = max(200, _bottom_total_w - (BOTTOM_CONTROLS_W + 2 * BOTTOM_COL_GAP))
    BOTTOM_INFO_W = max(140, _bottom_total_w - (_bottom_table_w + BOTTOM_CONTROLS_W + 2 * BOTTOM_COL_GAP))
BOTTOM_TABLE = (BOTTOM_PANEL[0], BOTTOM_PANEL[0] + _bottom_table_w, BOTTOM_PANEL[2], BOTTOM_PANEL[3])
BOTTOM_INFO = (BOTTOM_TABLE[1] + BOTTOM_COL_GAP, BOTTOM_TABLE[1] + BOTTOM_COL_GAP + BOTTOM_INFO_W, BOTTOM_PANEL[2], BOTTOM_PANEL[3])
BOTTOM_CONTROLS = (BOTTOM_INFO[1] + BOTTOM_COL_GAP, BOTTOM_PANEL[1], BOTTOM_PANEL[2], BOTTOM_PANEL[3])


# -------------------------
# Pygame drawing helpers
# -------------------------
class Colors:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    LIGHT_GRAY = (211, 211, 211)
    DARK_GRAY = (169, 169, 169)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)


FONT_SCALE = 1.35
FONT_MIN_SIZE = 11
FONT_CANDIDATES = (
    "Apple SD Gothic Neo",
    "AppleGothic",
    "Malgun Gothic",
    "MalgunGothic",
    "NanumGothic",
    "Nanum Gothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "Arial Unicode MS",
)


def _to_screen_y(y: float) -> float:
    return SCREEN_H - y


def draw_lrbt_rectangle_filled(surface, left, right, bottom, top, color):
    rect = pygame.Rect(left, _to_screen_y(top), right - left, top - bottom)
    if len(color) == 4 and color[3] < 255:
        temp = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        temp.fill(color)
        surface.blit(temp, rect.topleft)
    else:
        pygame.draw.rect(surface, color[:3], rect)


def draw_lrbt_rectangle_outline(surface, left, right, bottom, top, color, width=1):
    rect = pygame.Rect(left, _to_screen_y(top), right - left, top - bottom)
    pygame.draw.rect(surface, color[:3], rect, width)


def draw_line(surface, x1, y1, x2, y2, color, width=1):
    pygame.draw.line(
        surface,
        color[:3],
        (x1, _to_screen_y(y1)),
        (x2, _to_screen_y(y2)),
        width,
    )


def draw_dotted_line_horizontal_lrbt(
    surface, x_left, x_right, y, color, width=1, dash=4, gap=3
):
    """Horizontal reference in LRBT y (same convention as draw_line)."""
    x = float(x_left)
    x_right = float(x_right)
    while x < x_right:
        seg_end = min(x + dash, x_right)
        draw_line(surface, x, y, seg_end, y, color, width)
        x = seg_end + gap


def draw_line_strip(surface, points, color, width=1):
    if len(points) < 2:
        return
    pts = [(x, _to_screen_y(y)) for x, y in points]
    pygame.draw.lines(surface, color[:3], False, pts, width)


def draw_circle_filled(surface, x, y, radius, color):
    pygame.draw.circle(surface, color[:3], (int(x), int(_to_screen_y(y))), int(radius))


def draw_circle_outline(surface, x, y, radius, color, width=1):
    pygame.draw.circle(surface, color[:3], (int(x), int(_to_screen_y(y))), int(radius), width)


class SliderWidget:
    def __init__(self, x, y, width, height, min_value, max_value, value):
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.value = float(value)
        self.dragging = False

    def _norm_value(self) -> float:
        denom = max(1e-12, self.max_value - self.min_value)
        return (self.value - self.min_value) / denom

    def contains(self, mx, my) -> bool:
        return (self.x <= mx <= (self.x + self.width)) and (self.y <= my <= (self.y + self.height))

    def set_value_from_mouse(self, mx):
        t = (mx - self.x) / max(1e-12, self.width)
        t = max(0.0, min(1.0, t))
        self.value = self.min_value + t * (self.max_value - self.min_value)

    def handle_mouse_down(self, mx, my) -> bool:
        if self.contains(mx, my):
            self.dragging = True
            self.set_value_from_mouse(mx)
            return True
        return False

    def handle_mouse_up(self, mx, my):
        self.dragging = False

    def handle_mouse_motion(self, mx, my):
        if self.dragging:
            self.set_value_from_mouse(mx)

    def draw(self, surface):
        track_y = self.y + self.height * 0.5
        draw_line(surface, self.x, track_y, self.x + self.width, track_y, Colors.DARK_GRAY, 3)
        knob_radius = max(5, int(self.height * 0.6))
        knob_x = self.x + self._norm_value() * self.width
        draw_circle_filled(surface, knob_x, track_y, knob_radius, Colors.LIGHT_GRAY)
        draw_circle_outline(surface, knob_x, track_y, knob_radius, Colors.BLACK, 1)


class InputBox:
    def __init__(self, x, y, width, height, text="", font_size=11):
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)
        self.text = text
        self.font_size = font_size
        self.focused = False

    def contains(self, mx, my) -> bool:
        return (self.x <= mx <= (self.x + self.width)) and (self.y <= my <= (self.y + self.height))

    def clear_focus(self):
        self.focused = False

    def set_focus(self, focus: bool):
        self.focused = bool(focus)

    def handle_backspace(self):
        if self.text:
            self.text = self.text[:-1]

    def handle_text(self, text: str):
        self.text += text



# -------------------------
# Controls default (초기 상태 및 입자/피스톤 파라미터)
# -------------------------
DEFAULT_N =   1     # each for > 0 mol for <0   # AVOGADRO_NUMBER_SIMULATOR       # 초기 입자 수 (large for stable averages)
if DEFAULT_N < -1 * AVOGADRO_NUMBER_SIMULATOR or DEFAULT_N > 2:
    raise ValueError(f"DEFAULT_N must be between {-1 * AVOGADRO_NUMBER_SIMULATOR} and 2")

N_UNIT = "particles" if DEFAULT_N < 0 else "mols"
DEFAULT_N = abs(DEFAULT_N) if DEFAULT_N < 0 else int(DEFAULT_N * AVOGADRO_NUMBER_SIMULATOR)

ADJUSTED_KPA_PA = KPA_PA / PARTICLE_NUMBER_SCALE if N_UNIT == "mols" else KPA_PA


MAX_RENDER_PARTICLES = 200  # 렌더링에 사용할 최대 입자 수 (나머지는 계산만 참여)

BERENDSEN_TAU = 99999          # 목표 온도로 완만히 수렴하는 특성시간 (초, 물리시간)
BERENDSEN_MAX_ALPHA = 0.0  # 한 번 적용 시 dt/τ 상한 (강제 캡으로 과도 억제)
BERENDSEN_EVERY = 1          # 몇 프레임마다 한 번 적용 (1이면 매 프레임)
BGK_PRESERVE_ENERGY = False  # BGK는 분포만 혼합, 내부에너지는 보존 (large-N only)

if DEFAULT_N>MAX_RENDER_PARTICLES:
    # Pairwise stochastic thermostat (Lowe–Andersen style) to mix momentum at bath_Tk.
    # Dimensionless strength that scales physical collision freq ν = v_rms / λ_mfp into pair resampling rate.
    # Smaller is cheaper; 0 disables the thermostat.
    # BGK_STRENGTH = 5e-6 * DEFAULT_N/AVOGADRO_NUMBER_SIMULATOR
    COLLISION_RATE_SCALING = 0.5
    THERMAL_LAMBDA = -1
    BGK_PRESERVE_ENERGY = True

    # 느슨한 Berendsen velocity-rescaling (온도 요동 완화용, 분포 꼬리는 다소 억제)
    # https://en.wikipedia.org/wiki/Berendsen_thermostat
    BERENDSEN_TAU = 1. / COLLISION_RATE_SCALING         # 목표 온도로 완만히 수렴하는 특성시간 (초, 물리시간)
    BERENDSEN_MAX_ALPHA =   (1./60) /BERENDSEN_TAU * 1.2   # 한 번 적용 시 dt/τ 상한 (강제 캡으로 과도 억제)
else:
    # BGK_STRENGTH = -1
    COLLISION_RATE_SCALING = -1.0
    # Thermall emission strength: 0=elastic, 1=thermal emission (surface 혼합도)
    THERMAL_LAMBDA = 1.0
    BGK_PRESERVE_ENERGY = False





def get_app_dir() -> str:
    """
    Return directory where the executable resides.
    - PyInstaller onedir: directory containing CarnotApp.exe
    - Running as script: directory containing this .py file
    """
    if getattr(sys, "frozen", False):
        return str(Path(sys.executable).resolve().parent)
    return str(Path(__file__).resolve().parent)


def _set_windows_dpi_aware() -> None:
    """Use physical pixels on Windows so display size matches the taskbar work area."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _get_windows_work_area() -> Optional[Tuple[int, int, int, int]]:
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        rect = RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            return rect.left, rect.top, rect.right, rect.bottom
    except Exception:
        pass
    return None


def _available_display_size() -> Tuple[int, int]:
    """Max drawable window size that should stay above the taskbar."""
    pygame.display.init()
    info = pygame.display.Info()
    max_w = max(640, int(info.current_w))
    max_h = max(480, int(info.current_h))

    work = _get_windows_work_area()
    if work is not None:
        left, top, right, bottom = work
        max_w = max(640, right - left)
        max_h = max(480, bottom - top)

    # Small margin for window chrome / rounding
    return max(640, max_w - 8), max(480, max_h - 8)


def _fit_window_to_display(target_w: int, target_h: int) -> Tuple[int, int, float]:
    """Return (window_w, window_h, uniform_scale) fitting the usable desktop area."""
    avail_w, avail_h = _available_display_size()
    scale = min(1.0, avail_w / target_w, avail_h / target_h)
    window_w = max(640, int(round(target_w * scale)))
    window_h = max(480, int(round(target_h * scale)))
    scale = min(window_w / target_w, window_h / target_h)
    return window_w, window_h, scale


def playback_max_for_unit() -> float:
    return PLAYBACK_MAX_MOLS if N_UNIT == "mols" else PLAYBACK_MAX_PARTICLES


def playback_log_max_for_unit() -> float:
    return math.log10(playback_max_for_unit())


def is_fast_playback(scale: float) -> bool:
    return scale > FAST_MODE_THRESHOLD


def make_timestamped_log_paths(app_dir: str) -> Tuple[str, str]:
    stamp = datetime.now().strftime("%y%m%d_%H%M")
    return (
        os.path.join(app_dir, f"gas_log_{stamp}.xlsx"),
        os.path.join(app_dir, f"particle_log_{stamp}.xlsx"),
    )


class XlsxLogWriter:
    """Append rows to an .xlsx file; save periodically and on close."""

    def __init__(self, path: str, headers: List[str], save_every: int = 20):
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.path = path
        self.save_every = save_every
        self._rows_since_save = 0
        self._wb = Workbook()
        self._ws = self._wb.active
        self._ws.append(headers)
        self._save()

    def append(self, row: List):
        self._ws.append(row)
        self._rows_since_save += 1
        if self._rows_since_save >= self.save_every:
            self._save()
            self._rows_since_save = 0

    def set_save_every(self, n: int):
        self.save_every = max(1, int(n))

    def _save(self):
        try:
            self._wb.save(self.path)
        except Exception:
            pass

    def close(self):
        if self._wb is None:
            return
        self._save()
        self._wb = None
        self._ws = None


def _compute_pt_warmup_phys(n_sim: int) -> float:
    return TIME_AVERAGE + (n_sim - 1) * (3e-1 - TIME_AVERAGE) / (AVOGADRO_NUMBER_SIMULATOR - 1)


def _apply_particle_settings(n_sim: int, unit_mode: str):
    global DEFAULT_N, N_UNIT, ADJUSTED_KPA_PA
    global COLLISION_RATE_SCALING, THERMAL_LAMBDA, BERENDSEN_TAU, BERENDSEN_MAX_ALPHA
    global BGK_PRESERVE_ENERGY

    DEFAULT_N = int(n_sim)
    N_UNIT = unit_mode
    ADJUSTED_KPA_PA = KPA_PA / PARTICLE_NUMBER_SCALE if N_UNIT == "mols" else KPA_PA

    if DEFAULT_N > MAX_RENDER_PARTICLES:
        COLLISION_RATE_SCALING = 0.5
        THERMAL_LAMBDA = -1
        BGK_PRESERVE_ENERGY = True
        BERENDSEN_TAU = 1.0 / COLLISION_RATE_SCALING
        BERENDSEN_MAX_ALPHA = (1.0 / 60.0) / BERENDSEN_TAU * 1.2
    else:
        COLLISION_RATE_SCALING = -1.0
        THERMAL_LAMBDA = 1.0
        BGK_PRESERVE_ENERGY = False
        BERENDSEN_TAU = 99999
        BERENDSEN_MAX_ALPHA = 0.0


# piston (피스톤 초기값)
PISTON_Y_INIT = CUBIC_ROOT_OF_VOLUME_PX+PANEL_Y0         # 피스톤 초기 Y 위치(픽셀)
U_INIT = +0.0                 # 피스톤 속도 초기값 (픽셀/초, +면 팽창, -면 압축)
# PISTON_U_STEP_PX = 250       
# PISTON_U_STEP_M_S = PISTON_U_STEP_PX * SPEED_SCALE
PISTON_U_STEP_M_S = 0.001  # 키 입력 시 변화량 (m/s)
PISTON_U_STEP_PX = PISTON_U_STEP_M_S / SPEED_SCALE
CYLINDER_DEPTH_PX = CUBIC_ROOT_OF_VOLUME_PX     # 실린더 깊이(픽셀). 압력 계산 시 사용

# -------------------------
# Thermal velocity helpers
# -------------------------
def sigma_from_Tk(Tk: float) -> float:
    """Map Kelvin-ish UI to simulation velocity scale."""
    Tk = max(1e-6, Tk)
    sigma_m_s = math.sqrt(KB * Tk / MASS)
    return sigma_m_s / SPEED_SCALE


def sample_thermal_velocity_for_wall(Tk: float, wall: str) -> Tuple[float, float]:
    """
    Thermal wall emission in 2D:
      v_t ~ N(0, sigma^2)
      v_n > 0 with pdf proportional to v_n * exp(-v_n^2/(2 sigma^2))
      => v_n = sqrt(-2 sigma^2 ln(1-r))
    wall: 'L','R','B' (left/right/bottom)
    """
    sigma = sigma_from_Tk(Tk)

    r = random.random()
    vn = math.sqrt(-2.0 * sigma * sigma * math.log(max(1e-12, 1.0 - r)))
    vt = random.gauss(0.0, sigma)

    if wall == "L":
        vx, vy = +vn, vt
    elif wall == "R":
        vx, vy = -vn, vt
    elif wall == "B":
        vx, vy = vt, +vn
    else:
        raise ValueError("wall must be L/R/B")
    return vx, vy


def _sample_thermal_velocity_for_wall_vec_sigma(
    sigma: float, wall: str, size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Same as sample_thermal_velocity_for_wall_vec but with sigma precomputed."""
    r = np.random.random(size).astype(np.float32)
    vn = np.sqrt(-2.0 * (sigma * sigma) * np.log(np.maximum(1e-12, 1.0 - r))).astype(np.float32)
    vt = np.random.normal(0.0, sigma, size).astype(np.float32)
    if wall == "L":
        return +vn, vt
    if wall == "R":
        return -vn, vt
    if wall == "B":
        return vt, +vn
    raise ValueError("wall must be L/R/B")


def sample_thermal_velocity_for_wall_vec(Tk: float, wall: str, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized thermal emission for walls. Returns float32 arrays."""
    return _sample_thermal_velocity_for_wall_vec_sigma(sigma_from_Tk(Tk), wall, size)


# -------------------------
# Shared math helpers
# -------------------------
def _compute_v2_m(
    vx: np.ndarray,
    vy: np.ndarray,
    speed_scale: float,
    vz: Optional[np.ndarray] = None,
) -> np.ndarray:
    # In-place accumulation reduces N-sized temporaries (hot path for large N)
    v2 = vx * vx
    v2 += vy * vy
    if vz is not None:
        v2 += vz * vz
    return v2 * (speed_scale * speed_scale)


def _compute_temperature_from_v2(v2_m: np.ndarray, mass: float, dof: int = 3) -> float:
    return float(mass * np.mean(v2_m) / (float(dof) * KB))


def _compute_geometry_m(piston_y: float, clamp_height: bool = True) -> Tuple[float, float, float]:
    width_m = (SIM_X1 - SIM_X0) * PX_TO_M
    depth_m = CYLINDER_DEPTH_PX * PX_TO_M
    height_m = (piston_y - SIM_Y0) * PX_TO_M
    if clamp_height:
        height_m = max(1e-9, height_m)
    return width_m, height_m, depth_m


def _compute_volume_m3(piston_y: float, clamp_height: bool = True) -> float:
    width_m, height_m, depth_m = _compute_geometry_m(piston_y, clamp_height=clamp_height)
    return width_m * height_m * depth_m


def _compute_volume_L(piston_y: float, clamp_height: bool = True) -> float:
    return _compute_volume_m3(piston_y, clamp_height=clamp_height) * 1000.0


def _compute_area_tb_lr(piston_y: float) -> Tuple[float, float]:
    width_m, height_m, depth_m = _compute_geometry_m(piston_y, clamp_height=True)
    area_tb = width_m * depth_m
    area_lr = height_m * depth_m
    return area_tb, area_lr


# -------------------------
# Simulation state
# -------------------------
@dataclass
class SimState:
    N: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray

    piston_y: float
    piston_u: float  # wall speed u (px/s)

    thermal_on: bool   # True: thermal walls active for L/R/B; False: elastic
    bath_Tk: float     # single bath temperature used for all thermal walls (학생이 바꿈)

    # accumulators per step
    impulse_piston_y1: float = 0.0
    impulse_piston_y2: float = 0.0
    impulse_piston_x1: float = 0.0
    impulse_piston_x2: float = 0.0
    dQ_step: float = 0.0
    dW_by_step: float = 0.0


# -------------------------
# Diagnostics / logging
# -------------------------
class Diagnostics:
    def __init__(self, max_points: int = 1500):
        self.t = 0.0
        self.max_points = max_points
        self.pt_warmup_phys = _compute_pt_warmup_phys(DEFAULT_N) 
        self._pt_warmup_sum_pressure = 0.0
        self._pt_warmup_sum_temperature = 0.0
        self._pt_warmup_count = 0
        self.window_impulse_sum = 0.0
        self.window_dt_sum = 0.0
        self.window_temperature_sum = 0.0
        self.window_step_count = 0
        self._frame_count = 0
        self.pressure_hold = None
        self.temperature_hold = None

        self.ts: List[float] = []
        self.Qs: List[float] = []
        self.Ws: List[float] = []
        self.Ps: List[float] = []
        self.Ts: List[float] = []
        self.Vs: List[float] = []

        self.Q_total = 0.0
        self.W_by_total = 0.0

        # PT trajectory
        self.PT: List[Tuple[float, float]] = []

        # Particle mode metrics (windowed)
        self.vx_rms_s: List[float] = []
        self.vy_rms_s: List[float] = []
        self.vz_rms_s: List[float] = []
        self.f_top_s: List[float] = []
        self.window_top_impulse_sum = 0.0
        self.vx_rms_hold = 0.0
        self.vy_rms_hold = 0.0
        self.vz_rms_hold = 0.0
        self.f_top_hold = 0.0

        # speed histogram cache
        self.hist_bins = 40  # finer speed bins for clearer shape
        self.hist_vmax = HIST_VMAX_M_S
        self.hist_counts = np.zeros(self.hist_bins, dtype=float)
        # logging
        self.log_path_gas = None
        self.log_path_particle = None
        self._log_writer_gas: Optional[XlsxLogWriter] = None
        self._log_writer_particle: Optional[XlsxLogWriter] = None

    def _trim(self):
        if len(self.ts) > self.max_points:
            self.ts = self.ts[-self.max_points:]
            self.Qs = self.Qs[-self.max_points:]
            self.Ws = self.Ws[-self.max_points:]
            self.Ps = self.Ps[-self.max_points:]
            self.Ts = self.Ts[-self.max_points:]
            self.Vs = self.Vs[-self.max_points:]
            self.vx_rms_s = self.vx_rms_s[-self.max_points:]
            self.vy_rms_s = self.vy_rms_s[-self.max_points:]
            self.vz_rms_s = self.vz_rms_s[-self.max_points:]
            self.f_top_s = self.f_top_s[-self.max_points:]
        if len(self.PT) > self.max_points:
            self.PT = self.PT[-self.max_points:]

    def set_log_paths(self, gas_path: str, particle_path: str):
        if self._log_writer_gas:
            try:
                self._log_writer_gas.close()
            except Exception:
                pass
        if self._log_writer_particle:
            try:
                self._log_writer_particle.close()
            except Exception:
                pass
        self.log_path_gas = gas_path
        self.log_path_particle = particle_path
        save_every = XLSX_SAVE_EVERY
        if gas_path:
            self._log_writer_gas = XlsxLogWriter(
                gas_path,
                ["time_s", "P_kPa", "V_L", "T_K", "Q_J", "W_J"],
                save_every=save_every,
            )
        else:
            self._log_writer_gas = None
        if particle_path:
            self._log_writer_particle = XlsxLogWriter(
                particle_path,
                ["time_s", "vx_rms_m_s", "vy_rms_m_s", "vz_rms_m_s", "f_top_N"],
                save_every=save_every,
            )
        else:
            self._log_writer_particle = None

    def set_log_save_every(self, save_every: int):
        n = max(1, int(save_every))
        if self._log_writer_gas:
            self._log_writer_gas.set_save_every(n)
        if self._log_writer_particle:
            self._log_writer_particle.set_save_every(n)

    def update(self, state: SimState, dt_phys: float, hist_every: int = HIST_EVERY):
        dt_phys = max(1e-12, dt_phys)
        self.t += dt_phys

        # accumulate
        self.Q_total += state.dQ_step 
        self.W_by_total += state.dW_by_step 

        # Temperature from mean v^2 (3D):
        # (1/2)m <v^2> = (d/2) kT with d=3 -> T = m <v^2> / (3k)
        v2_m = _compute_v2_m(state.vx, state.vy, SPEED_SCALE, vz=state.vz)
        T_sim = _compute_temperature_from_v2(v2_m, MASS_F32, dof=3)

        # Pressure from impulse on walls during dt (average of 4 sides):
        area_tb, area_lr = _compute_area_tb_lr(state.piston_y)
        imp_per_area_top = abs(state.impulse_piston_y2)  / area_tb
        imp_per_area_bottom = abs(state.impulse_piston_y1)  / area_tb
        imp_per_area_left = abs(state.impulse_piston_x1)  / area_lr
        imp_per_area_right = abs(state.impulse_piston_x2)  / area_lr
        # self.window_impulse_sum += (+imp_per_area_left+imp_per_area_right)/2
        # self.window_impulse_sum += (imp_per_area_bottom)
        # self.window_impulse_sum += (imp_per_area_top)

        self.window_impulse_sum += (imp_per_area_top+imp_per_area_bottom+imp_per_area_left+imp_per_area_right)/4
        self.window_top_impulse_sum += abs(state.impulse_piston_y2)

        self.window_dt_sum += dt_phys
        self.window_temperature_sum += T_sim
        self.window_step_count += 1
        if self.window_dt_sum >= self.pt_warmup_phys:
            P_window = abs(self.window_impulse_sum) / (self.window_dt_sum)
            T_window = self.window_temperature_sum / self.window_step_count
            self.pressure_hold = P_window
            self.temperature_hold = T_window
            if state.vx.size > 0:
                # sqrt(<v_px^2>) * SPEED_SCALE == sqrt(<v_m_s^2>); avoids three N-sized scaled arrays.
                self.vx_rms_hold = float(np.sqrt(np.mean(state.vx * state.vx))) * SPEED_SCALE
                self.vy_rms_hold = float(np.sqrt(np.mean(state.vy * state.vy))) * SPEED_SCALE
                self.vz_rms_hold = float(np.sqrt(np.mean(state.vz * state.vz))) * SPEED_SCALE
            else:
                self.vx_rms_hold = 0.0
                self.vy_rms_hold = 0.0
                self.vz_rms_hold = 0.0
            self.f_top_hold = self.window_top_impulse_sum / self.window_dt_sum
            self.window_impulse_sum = 0.0
            self.window_top_impulse_sum = 0.0
            self.window_dt_sum = 0.0
            self.window_temperature_sum = 0.0
            self.window_step_count = 0
            P_use = self.pressure_hold
            T_use = self.temperature_hold
            volume_L = _compute_volume_L(state.piston_y, clamp_height=True)
            self.ts.append(self.t)
            self.Qs.append(self.Q_total)
            self.Ws.append(self.W_by_total)
            self.PT.append((P_use, T_use))
            self.Ps.append(P_use)
            self.Ts.append(T_use)
            self.Vs.append(volume_L)
            self.vx_rms_s.append(self.vx_rms_hold)
            self.vy_rms_s.append(self.vy_rms_hold)
            self.vz_rms_s.append(self.vz_rms_hold)
            self.f_top_s.append(self.f_top_hold)
            # log to XLSX if enabled
            if self._log_writer_gas:
                p_kpa = P_use / ADJUSTED_KPA_PA
                try:
                    self._log_writer_gas.append(
                        [self.t, p_kpa, volume_L, T_use, self.Q_total, self.W_by_total]
                    )
                except Exception:
                    pass
            if self._log_writer_particle:
                try:
                    self._log_writer_particle.append(
                        [
                            self.t,
                            self.vx_rms_hold,
                            self.vy_rms_hold,
                            self.vz_rms_hold,
                            self.f_top_hold,
                        ]
                    )
                except Exception:
                    pass
            # Only the appending path grows the lists, so trim here instead of every call.
            self._trim()

        # histogram (throttled)
        self._frame_count += 1
        if (self._frame_count % max(1, hist_every)) == 0:
            speeds_m_s = np.sqrt(v2_m)
            counts, _ = np.histogram(speeds_m_s, bins=self.hist_bins, range=(0.0, self.hist_vmax))
            self.hist_counts = counts.astype(float)

    def reset(self, reset_logs: bool = False, log_paths: Optional[Tuple[str, str]] = None):
        self.t = 0.0
        self.ts.clear()
        self.Qs.clear()
        self.Ws.clear()
        self.Ps.clear()
        self.Ts.clear()
        self.Vs.clear()
        self.PT.clear()
        self.vx_rms_s.clear()
        self.vy_rms_s.clear()
        self.vz_rms_s.clear()
        self.f_top_s.clear()
        self.window_top_impulse_sum = 0.0
        self.vx_rms_hold = 0.0
        self.vy_rms_hold = 0.0
        self.vz_rms_hold = 0.0
        self.f_top_hold = 0.0
        self.Q_total = 0.0
        self.W_by_total = 0.0
        self.hist_counts[:] = 0.0
        self._pt_warmup_sum_pressure = 0.0
        self._pt_warmup_sum_temperature = 0.0
        self._pt_warmup_count = 0
        self.window_impulse_sum = 0.0
        self.window_dt_sum = 0.0
        self.window_temperature_sum = 0.0
        self.window_step_count = 0
        if reset_logs:
            if self._log_writer_gas:
                try:
                    self._log_writer_gas.close()
                except Exception:
                    pass
            if self._log_writer_particle:
                try:
                    self._log_writer_particle.close()
                except Exception:
                    pass
            self._log_writer_gas = None
            self._log_writer_particle = None
            if log_paths is not None:
                gas_path, particle_path = log_paths
            else:
                gas_path, particle_path = self.log_path_gas, self.log_path_particle
            if gas_path or particle_path:
                self.set_log_paths(gas_path, particle_path)


# -------------------------
# Physics engine
# -------------------------
class Engine:
    # -------------------------
    # Main integration step
    # -------------------------
    def step(self, state: SimState, sub_dt: float):
        # clear step accumulators
        state.impulse_piston_y1 = 0.0
        state.impulse_piston_y2 = 0.0
        state.impulse_piston_x1 = 0.0
        state.impulse_piston_x2 = 0.0
        state.dQ_step = 0.0
        state.dW_by_step = 0.0

        lam = min(1.0, THERMAL_LAMBDA) if state.thermal_on else -1
        # Compute energy sum without materializing scaled v^2 array
        sum_v2_start = float(
            np.sum(state.vx * state.vx)
            + np.sum(state.vy * state.vy)
            + np.sum(state.vz * state.vz)
        )
        e_coeff = 0.5 * MASS_F32 * (SPEED_SCALE_F32 * SPEED_SCALE_F32)
        E_start = e_coeff * sum_v2_start

        xmin, xmax = SIM_X0, SIM_X1
        ymin = SIM_Y0
        width = xmax - xmin
        piston_y0 = state.piston_y
        u = state.piston_u
        lower_limit = SIM_Y0 + 100
        upper_limit = SIM_Y1
        piston_mid = piston_y0 + u * sub_dt * 0.5
        if piston_mid <= lower_limit:
            piston_mid = float(lower_limit)
        elif piston_mid >= upper_limit:
            piston_mid = float(upper_limit)
        height_ref = piston_mid - ymin

        n_L, n_R = self._handle_x_collisions(state, sub_dt, xmin, xmax, width)
        n_B, n_T, bottom_vn_in = self._handle_y_collisions(state, sub_dt, ymin, piston_mid, height_ref, u)
        self._handle_z_collisions(state, sub_dt, 0.0, CYLINDER_DEPTH_PX, CYLINDER_DEPTH_PX)

        sum_v2_end = float(
            np.sum(state.vx * state.vx)
            + np.sum(state.vy * state.vy)
            + np.sum(state.vz * state.vz)
        )
        E_det_end = e_coeff * sum_v2_end
        state.dW_by_step = float(-(E_det_end - E_start))

        self._apply_thermal_walls(state, n_L, n_R, n_B, bottom_vn_in, lam)
        self._advance_piston(state, sub_dt, lower_limit, upper_limit)

        # fold any residual penetrations exactly via mirrored tiling (energy/momentum-consistent)
        # self._sweep_reflect(state)

    # -------------------------
    # X-axis collisions
    # -------------------------
    def _handle_x_collisions(
        self,
        state: SimState,
        sub_dt: float,
        xmin: float,
        xmax: float,
        width: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 1) X multi-bounce against fixed walls
        # state.x/vx are read-only here until the final in-place write below,
        # so we can avoid the upfront copy.
        x0 = state.x
        vx0 = state.vx
        abs_vx = np.abs(vx0)
        t_first = np.full_like(vx0, np.inf)
        mask_pos = vx0 > 0
        mask_neg = vx0 < 0
        if np.any(mask_pos):
            t_first[mask_pos] = (xmax - x0[mask_pos]) / vx0[mask_pos]
        if np.any(mask_neg):
            t_first[mask_neg] = (x0[mask_neg] - xmin) / (-vx0[mask_neg])

        period = np.full_like(vx0, np.inf)
        moving = abs_vx > 0
        if np.any(moving):
            period[moving] = width / abs_vx[moving]

        n_hits = np.zeros_like(vx0, dtype=np.int32)
        hit_mask = t_first <= sub_dt
        if np.any(hit_mask):
            n_hits[hit_mask] = (1 + np.floor((sub_dt - t_first[hit_mask]) / period[hit_mask])).astype(np.int32)

        n_L = np.zeros_like(n_hits)
        n_R = np.zeros_like(n_hits)
        if np.any(mask_pos):
            n_R[mask_pos] = (n_hits[mask_pos] + 1) // 2
            n_L[mask_pos] = n_hits[mask_pos] // 2
        if np.any(mask_neg):
            n_L[mask_neg] = (n_hits[mask_neg] + 1) // 2
            n_R[mask_neg] = n_hits[mask_neg] // 2

        # mirrored tiling for final x position
        x_rel = x0 - xmin
        period_pos = 2.0 * width
        x_unwrapped = x_rel + vx0 * sub_dt
        x_mod = np.mod(x_unwrapped, period_pos)
        x_rel_final = np.where(x_mod > width, period_pos - x_mod, x_mod)
        state.x[:] = xmin + x_rel_final

        flip = (n_hits & 1) != 0
        state.vx[:] = np.where(flip, -vx0, vx0)

        abs_vx_m = abs_vx * SPEED_SCALE_F32
        if np.any(n_L):
            state.impulse_piston_x1 += float(np.sum(MASS_F32 * (-2.0 * abs_vx_m) * n_L))
        if np.any(n_R):
            state.impulse_piston_x2 += float(np.sum(MASS_F32 * (2.0 * abs_vx_m) * n_R))

        return n_L, n_R

    # -------------------------
    # Z-axis collisions (depth)
    # -------------------------
    def _handle_z_collisions(
        self,
        state: SimState,
        sub_dt: float,
        zmin: float,
        zmax: float,
        depth: float,
    ) -> None:
        if depth <= 0.0:
            state.z[:] = zmin
            state.vz[:] = 0.0
            return

        # state.z/vz are read-only here until the final in-place writes below.
        z0 = state.z
        vz0 = state.vz
        abs_vz = np.abs(vz0)

        t_first = np.full_like(vz0, np.inf)
        mask_pos = vz0 > 0
        mask_neg = vz0 < 0
        if np.any(mask_pos):
            t_first[mask_pos] = (zmax - z0[mask_pos]) / vz0[mask_pos]
        if np.any(mask_neg):
            t_first[mask_neg] = (z0[mask_neg] - zmin) / (-vz0[mask_neg])

        period = np.full_like(vz0, np.inf)
        moving = abs_vz > 0
        if np.any(moving):
            period[moving] = depth / abs_vz[moving]

        n_hits = np.zeros_like(vz0, dtype=np.int32)
        hit_mask = t_first <= sub_dt
        if np.any(hit_mask):
            n_hits[hit_mask] = (
                1 + np.floor((sub_dt - t_first[hit_mask]) / period[hit_mask])
            ).astype(np.int32)

        z_rel = z0 - zmin
        period_pos = 2.0 * depth
        z_unwrapped = z_rel + vz0 * sub_dt
        z_mod = np.mod(z_unwrapped, period_pos)
        z_rel_final = np.where(z_mod > depth, period_pos - z_mod, z_mod)
        state.z[:] = zmin + z_rel_final

        flip = (n_hits & 1) != 0
        state.vz[:] = np.where(flip, -vz0, vz0)

    # -------------------------
    # Y-axis collisions
    # -------------------------
    def _handle_y_collisions(
        self,
        state: SimState,
        sub_dt: float,
        ymin: float,
        piston_mid: float,
        height_ref: float,
        u: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 2) Y vectorized multi-bounce with fixed top (approx)
        # vy0 must be a copy because state.vy is written before its later reads;
        # y0 is only read prior to state.y being written, so a reference is safe.
        y0 = state.y
        vy0 = state.vy.copy()
        n_B = np.zeros_like(vy0, dtype=np.int32)
        n_T = np.zeros_like(n_B)
        bottom_vn_in = np.zeros_like(vy0)

        if height_ref <= 1e-9:
            state.y[:] = ymin
            state.vy[:] = 0.0
            return n_B, n_T, bottom_vn_in

        abs_vy = np.abs(vy0)
        t_first_y = np.full_like(vy0, np.inf)
        mask_up = vy0 > 0
        mask_down = vy0 < 0
        if np.any(mask_up):
            t_first_y[mask_up] = (piston_mid - y0[mask_up]) / vy0[mask_up]
        if np.any(mask_down):
            t_first_y[mask_down] = (y0[mask_down] - ymin) / (-vy0[mask_down])
        t_first_y = np.maximum(t_first_y, 0.0)

        period_y = np.full_like(vy0, np.inf)
        moving_y = abs_vy > 0
        if np.any(moving_y):
            period_y[moving_y] = height_ref / abs_vy[moving_y]

        # Single allocation: n_hits_y holds rough values until the corrections below
        # overwrite specific indices. Fancy indexing into n_hits_y already returns
        # copies, so the "rough" snapshot does not need a separate full array.
        n_hits_y = np.zeros_like(vy0, dtype=np.int32)
        hit_mask_y = t_first_y <= sub_dt
        if np.any(hit_mask_y):
            n_hits_y[hit_mask_y] = (
                1 + np.floor((sub_dt - t_first_y[hit_mask_y]) / period_y[hit_mask_y])
            ).astype(np.int32)

            # Large-collision correction (harmonic/log approximation) for u>0
            if u > 0.0:
                LARGE_HITS = 6
                large_mask = hit_mask_y & moving_y & (n_hits_y >= LARGE_HITS)
                if np.any(large_mask):
                    idx = np.where(large_mask)[0]
                    v0 = abs_vy[idx]
                    t_first = t_first_y[idx]
                    n_hits_r = n_hits_y[idx]  # snapshot via fancy indexing
                    start_up = mask_up[idx]
                    n_T_est = np.where(start_up, (n_hits_r + 1) // 2, n_hits_r // 2)

                    eps = 1e-12
                    n_T_max = np.floor((v0 - eps) / (2.0 * u)).astype(np.int32)
                    n_T_max = np.maximum(n_T_max, 0)
                    n_T_eff = np.minimum(n_T_est, n_T_max)

                    valid = n_T_eff > 0
                    if np.any(valid):
                        v0_v = v0[valid]
                        n_T_v = n_T_eff[valid].astype(np.float64)
                        v_end = np.maximum(v0_v - 2.0 * u * n_T_v, eps)
                        log_term = np.log(v0_v / v_end)
                        log_ok = log_term > 1e-12
                        if np.any(log_ok):
                            v_eff = (2.0 * u * n_T_v[log_ok]) / log_term[log_ok]
                            period_eff = height_ref / v_eff
                            n_hits_corr = (
                                1 + np.floor((sub_dt - t_first[valid][log_ok]) / period_eff)
                            ).astype(np.int32)
                            n_hits_y[idx[valid][log_ok]] = np.maximum(n_hits_corr, 0)

                # Top catch-up clamp: limit top hits when piston outruns particle
                eps = 1e-12
                n_T_max_all = np.floor((abs_vy - eps) / (2.0 * u)).astype(np.int32)
                n_T_max_all = np.maximum(n_T_max_all, 0)
                if np.any(mask_up):
                    n_hits_y[mask_up] = np.minimum(n_hits_y[mask_up], 2 * n_T_max_all[mask_up])
                if np.any(mask_down):
                    n_hits_y[mask_down] = np.minimum(n_hits_y[mask_down], 2 * n_T_max_all[mask_down] + 1)

        if np.any(mask_up):
            n_T[mask_up] = (n_hits_y[mask_up] + 1) // 2
            n_B[mask_up] = n_hits_y[mask_up] // 2
        if np.any(mask_down):
            n_B[mask_down] = (n_hits_y[mask_down] + 1) // 2
            n_T[mask_down] = n_hits_y[mask_down] // 2

        # position update via mirrored tiling in [ymin, piston_mid]
        y_rel = y0 - ymin
        period_pos_y = 2.0 * height_ref
        y_unwrapped = y_rel + vy0 * sub_dt
        y_mod = np.mod(y_unwrapped, period_pos_y)
        y_rel_final = np.where(y_mod > height_ref, period_pos_y - y_mod, y_mod)
        state.y[:] = ymin + y_rel_final

        # closed-form kinematics for vy
        vy_final = vy0.copy()
        start_top = mask_up
        start_bottom = mask_down
        n_T_f = n_T.astype(np.float32)
        n_B_f = n_B.astype(np.float32)

        if np.any(start_top):
            k = np.minimum(n_T_f[start_top], n_B_f[start_top])
            v_pairs = vy0[start_top] - 2.0 * u * k
            vy_final[start_top] = np.where(
                n_T[start_top] > n_B[start_top],
                2.0 * u - v_pairs,
                v_pairs,
            )
        if np.any(start_bottom):
            k = n_T_f[start_bottom]
            v_pairs = vy0[start_bottom] + 2.0 * u * k
            vy_final[start_bottom] = np.where(
                n_B[start_bottom] > n_T[start_bottom],
                -v_pairs,
                v_pairs,
            )

        state.vy[:] = vy_final

        # bottom impulse (elastic bookkeeping)
        sum_v_in_B = np.zeros_like(vy0)
        if np.any(start_top):
            sum_v_in_B[start_top] = (
                u * n_B_f[start_top] * (n_B_f[start_top] + 1.0)
                - n_B_f[start_top] * vy0[start_top]
            )
        if np.any(start_bottom):
            sum_v_in_B[start_bottom] = (
                n_B_f[start_bottom] * vy0[start_bottom]
                + u * n_B_f[start_bottom] * (n_B_f[start_bottom] - 1.0)
            )
        if np.any(n_B):
            state.impulse_piston_y1 += float(
                np.sum(MASS_F32 * (2.0 * SPEED_SCALE_F32) * sum_v_in_B)
            )

        # top piston impulse (moving wall)
        if np.any(n_T):
            v_up0 = np.abs(vy0)
            impulse_top = n_T_f * v_up0 - u * n_T_f * n_T_f
            state.impulse_piston_y2 += float(
                np.sum(MASS_F32 * (2.0 * SPEED_SCALE_F32) * impulse_top)
            )

        # approximate last bottom incoming normal speed for thermal correction
        mask_top_B = start_top & (n_B > 0)
        if np.any(mask_top_B):
            bottom_vn_in[mask_top_B] = np.maximum(
                0.0, vy0[mask_top_B] - 2.0 * u * n_B_f[mask_top_B]
            )
        mask_bottom_B = start_bottom & (n_B > 0)
        if np.any(mask_bottom_B):
            bottom_vn_in[mask_bottom_B] = np.maximum(
                0.0, -(vy0[mask_bottom_B] + 2.0 * u * (n_B_f[mask_bottom_B] - 1.0))
            )

        return n_B, n_T, bottom_vn_in

    # -------------------------
    # Thermal wall interactions
    # -------------------------
    def _apply_thermal_walls(
        self,
        state: SimState,
        n_L: np.ndarray,
        n_R: np.ndarray,
        n_B: np.ndarray,
        bottom_vn_in: np.ndarray,
        lam: float,
    ):
        # 3) End-of-sub_dt thermalization (L/R/B only)
        if lam <= 0.0:
            return
        n_heat = n_L + n_R + n_B
        heat_mask = n_heat > 0
        if np.any(heat_mask):
            p_th = 1.0 - np.power((1.0 - lam), n_heat)
            rand = np.random.random(n_heat.shape)
            thermalize = heat_mask & (rand < p_th)
            if np.any(thermalize):
                idx = np.where(thermalize)[0]
                nL = n_L[idx]
                nR = n_R[idx]
                nB = n_B[idx]
                total = nL + nR + nB
                pick = np.random.random(idx.size) * total
                pick_L = pick < nL
                pick_R = (pick >= nL) & (pick < (nL + nR))
                pick_B = ~(pick_L | pick_R)

                vx_new = np.empty(idx.size, dtype=state.vx.dtype)
                vy_new = np.empty(idx.size, dtype=state.vy.dtype)
                sigma = sigma_from_Tk(state.bath_Tk)
                vz_new = np.random.normal(0.0, sigma, size=idx.size).astype(state.vx.dtype)

                if np.any(pick_L):
                    vx_th, vy_th = _sample_thermal_velocity_for_wall_vec_sigma(
                        sigma, "L", int(np.count_nonzero(pick_L))
                    )
                    vx_new[pick_L] = vx_th
                    vy_new[pick_L] = vy_th
                if np.any(pick_R):
                    vx_th, vy_th = _sample_thermal_velocity_for_wall_vec_sigma(
                        sigma, "R", int(np.count_nonzero(pick_R))
                    )
                    vx_new[pick_R] = vx_th
                    vy_new[pick_R] = vy_th
                if np.any(pick_B):
                    vx_th, vy_th = _sample_thermal_velocity_for_wall_vec_sigma(
                        sigma, "B", int(np.count_nonzero(pick_B))
                    )
                    vx_new[pick_B] = vx_th
                    vy_new[pick_B] = vy_th

                v0x_m = state.vx[idx] * SPEED_SCALE_F32
                v0y_m = state.vy[idx] * SPEED_SCALE_F32
                v0z_m = state.vz[idx] * SPEED_SCALE_F32
                v1x_m = vx_new * SPEED_SCALE_F32
                v1y_m = vy_new * SPEED_SCALE_F32
                v1z_m = vz_new * SPEED_SCALE_F32
                E0 = 0.5 * MASS_F32 * (v0x_m * v0x_m + v0y_m * v0y_m + v0z_m * v0z_m)
                E1 = 0.5 * MASS_F32 * (v1x_m * v1x_m + v1y_m * v1y_m + v1z_m * v1z_m)
                state.dQ_step += float(np.sum(E1 - E0))

                abs_vx_m = np.abs(v0x_m)
                if np.any(pick_L):
                    v_in_m = -abs_vx_m[pick_L]
                    v_out_m = v1x_m[pick_L]
                    elastic_imp = 2.0 * MASS_F32 * v_in_m
                    thermal_imp = MASS_F32 * (v_in_m - v_out_m)
                    state.impulse_piston_x1 += float(np.sum(thermal_imp - elastic_imp))
                if np.any(pick_R):
                    v_in_m = abs_vx_m[pick_R]
                    v_out_m = v1x_m[pick_R]
                    elastic_imp = 2.0 * MASS_F32 * v_in_m
                    thermal_imp = MASS_F32 * (v_in_m - v_out_m)
                    state.impulse_piston_x2 += float(np.sum(thermal_imp - elastic_imp))
                if np.any(pick_B):
                    vn_in_px = bottom_vn_in[idx][pick_B]
                    fallback = np.abs(state.vy[idx][pick_B])
                    vn_in_px = np.where(vn_in_px > 0.0, vn_in_px, fallback)
                    v_in_m = -vn_in_px * SPEED_SCALE_F32
                    v_out_m = v1y_m[pick_B]
                    elastic_imp = 2.0 * MASS_F32 * v_in_m
                    thermal_imp = MASS_F32 * (v_in_m - v_out_m)
                    state.impulse_piston_y1 += float(np.sum(thermal_imp - elastic_imp))

                state.vx[idx] = vx_new
                state.vy[idx] = vy_new
                state.vz[idx] = vz_new

    # -------------------------
    # Piston advance
    # -------------------------
    def _advance_piston(self, state: SimState, sub_dt: float, lower_limit: float, upper_limit: float):
        # 4) advance piston after this substep
        piston_y_next = state.piston_y + state.piston_u * sub_dt
        if piston_y_next <= lower_limit:
            piston_y_next = float(lower_limit)
            state.piston_u = 0.0
        elif piston_y_next >= upper_limit:
            piston_y_next = float(upper_limit)
            state.piston_u = 0.0
        state.piston_y = piston_y_next


# -------------------------
# UI / simulation app
# -------------------------
class CarnotApp:
    def __init__(self):
        _set_windows_dpi_aware()
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        pygame.init()
        pygame.font.init()
        win_w, win_h, window_scale = _fit_window_to_display(SCREEN_W, SCREEN_H)
        if window_scale >= 1.0:
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            self._display = None
            self._window_scale = 1.0
        else:
            self.screen = pygame.Surface((SCREEN_W, SCREEN_H))
            self._display = pygame.display.set_mode((win_w, win_h))
            self._window_scale = window_scale
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()

        self.engine = Engine()
        self.diag = Diagnostics()

        self.running = True
        self.playback_scale = PLAYBACK_SCALE
        self.playback_label_text = ""
        self._font_cache = {}
        self._font_path = self._resolve_font_path()
        self._text_cache = {}
        self._refresh_playback_label()

        # auto-pause targets
        self.target_T = -1.0
        self.target_P = -1.0
        self.target_V = -1.0
        # auto-pause crossing detection (stores last checked values)
        self._prev_auto_T = None
        self._prev_auto_P = None
        self._prev_auto_V = None
        self.target_inputs = {
            "T": InputBox(0, 0, 70, 18, text="-1", font_size=11),
            "P": InputBox(0, 0, 70, 18, text="-1", font_size=11),
            "V": InputBox(0, 0, 70, 18, text="-1", font_size=11),
        }
        self._focused_input_key = None

        self.playback_slider = SliderWidget(
            0,
            0,
            220,
            16,
            min_value=0.0,
            max_value=1.0,
            value=self._slider_from_playback(self.playback_scale),
        )
        self._layout_controls()

        self.state = self._init_state(DEFAULT_N)
        self._substeps = 1  # 안정성: 한 프레임을 최소 2번 나눠서 적분
        self._frame_id = 0  # draw throttling
        self._force_draw = True
        self._fast_mode_active = False
        self._reset_pending_diag()
        self._graph_ts_np = None
        self._graph_Q_np = None
        self._graph_W_np = None
        # XLSX log path (cross-platform friendly); new pair on each reset
        self.diag.set_log_paths(*make_timestamped_log_paths(get_app_dir()))
        self._sync_log_save_interval()
        # particle-count prompt state (shown on start/reset)
        self._prompt_active = False
        self._prompt_text = ""
        self._prompt_error = ""
        self._prompt_mode = "mols" if N_UNIT == "mols" else "particles"
        self._prompt_reason = ""
        self._running_before_prompt = self.running
        self._begin_particle_prompt("start", resume_running=False)

    def _get_font(self, font_size: int):
        size = max(FONT_MIN_SIZE, int(round(font_size * FONT_SCALE)))
        cache_key = (self._font_path, size)
        font = self._font_cache.get(cache_key)
        if font is None:
            if self._font_path:
                font = pygame.font.Font(self._font_path, size)
            else:
                font = pygame.font.Font(None, size)
            self._font_cache[cache_key] = font
        return font

    def _resolve_font_path(self) -> Optional[str]:
        for name in FONT_CANDIDATES:
            try:
                path = pygame.font.match_font(name)
            except Exception:
                path = None
            if path:
                return path
        return None

    def _measure_text(self, text: str, font_size: int) -> Tuple[int, int]:
        font = self._get_font(font_size)
        return font.size(text)

    def _layout_controls(self):
        controls_margin = 25
        controls_x0, controls_x1, controls_y0, controls_y1 = BOTTOM_CONTROLS
        slider_w = 220
        slider_h = 16
        label_size = 11
        title_size = 11

        playback_title = "Simulation Time Scale (log)"
        playback_label_placeholder = "1 real s = 1.0e+00 sim s"
        target_title = "Auto-pause targets (-1=off)"
        row_labels = {"T": "T(K):", "P": "P(kPa):", "V": "V(L):"}
        num_target_rows = len(row_labels)


        vertical_gap_small = 4
        vertical_gap_pb_target = 60

        # whole playback panel width and height
        pb_title_w, pb_title_h = self._measure_text(playback_title, title_size)
        pb_label_w, pb_label_h = self._measure_text(playback_label_placeholder, label_size)
        pb_width = max(slider_w, pb_title_w, pb_label_w)
        pb_height = pb_title_h + vertical_gap_small + slider_h + vertical_gap_small + pb_label_h


        # whole target panel width and height
        row_label_w = 0
        row_label_h = 0
        for label in row_labels.values():
            w, h = self._measure_text(label, label_size)
            row_label_w = max(row_label_w, w)
            row_label_h = max(row_label_h, h)
        input_w = 70
        input_font_h = self._get_font(label_size).get_linesize()
        input_h = max(22, input_font_h + 6)
        row_h = max(row_label_h, input_h)
        row_gap = 6
        row_w = row_label_w + 6 + input_w
        target_title_w, target_title_h = self._measure_text(target_title, label_size)
        target_width = max(target_title_w, row_w)
        target_height = target_title_h + 4 + (row_h * num_target_rows) + (vertical_gap_small * num_target_rows)


        # whole contents in the control panel anchored to the bottom left corner
        stack_width = max(pb_width, target_width)
        stack_height = target_height + vertical_gap_pb_target + pb_height
        stack_right = controls_x1 - controls_margin ## JHS
        stack_bottom = controls_y1 - stack_height - 5 - controls_margin #+ controls_margin
        # stack_bottom = controls_y0 + controls_margin # baseline of contents in the control panel
        stack_left = stack_right - stack_width

        # position of playback title and slider
        self._playback_title_text = playback_title
        self._playback_title_pos = (stack_left, stack_bottom + target_height + vertical_gap_pb_target + pb_height)
        slider_top = (stack_bottom + target_height + vertical_gap_pb_target + pb_height) - pb_title_h - vertical_gap_small
        slider_bottom = slider_top - slider_h
        self.playback_slider.x = stack_left
        self.playback_slider.y = slider_bottom
        self.playback_slider.width = slider_w
        self.playback_slider.height = slider_h
        self._playback_label_pos = (stack_left, slider_bottom - 4)

        # position of target title and rows
        self._target_title_text = target_title
        self._target_title_pos = (stack_left, stack_bottom + target_height)
        first_row_top = (stack_bottom + target_height) - target_title_h - vertical_gap_small
        label_x = stack_left
        input_x = stack_left + row_label_w + 6
        self._target_row_labels = row_labels
        self._target_row_positions = {}
        for i, key in enumerate(("T", "P", "V")):
            row_top = first_row_top - i * (row_h + vertical_gap_small)
            row_center = row_top - row_h * 0.5
            row_bottom = row_top - row_h
            self._target_row_positions[key] = (label_x, row_center)
            box = self.target_inputs[key]
            box.x = input_x
            box.y = row_bottom #row_center - input_h * 0.5
            box.width = input_w
            box.height = row_h #input_h

    def _set_input_focus(self, key: Optional[str]):
        self._focused_input_key = key
        for k, box in self.target_inputs.items():
            box.set_focus(k == key)

    def _sync_targets_from_inputs(self):
        """Robustly sync target_T/P/V from UIInputText.text.

        Some Arcade versions/widgets don't reliably fire on_change per keystroke (or event payload differs),
        so we treat the input boxes as the source of truth.
        """
        for k, widget in self.target_inputs.items():
            txt = getattr(widget, "text", "")
            try:
                val = float(txt)
            except Exception:
                val = -1.0
            if k == "T":
                self.target_T = val
            elif k == "P":
                self.target_P = val
            elif k == "V":
                self.target_V = val

    # -------------------------
    # Playback helpers
    # -------------------------
    def _effective_intervals(self) -> dict:
        if is_fast_playback(self.playback_scale):
            return {
                "draw_every": FAST_DRAW_EVERY,
                "hist_every": FAST_HIST_EVERY,
                "graph_every": FAST_GRAPH_EVERY,
                "table_every": FAST_TABLE_EVERY,
                "diag_every": FAST_DIAG_EVERY,
            }
        return {
            "draw_every": 1,
            "hist_every": HIST_EVERY,
            "graph_every": 1,
            "table_every": TABLE_EVERY,
            "diag_every": 1,
        }

    def _maybe_refresh_graph_cache(self, force: bool = False):
        intervals = self._effective_intervals()
        if (
            not force
            and self._graph_ts_np is not None
            and (self._frame_id % intervals["graph_every"]) != 0
        ):
            return
        # diag.Qs/Ws are lists of scalar floats; vectorize the conversion + scaling.
        self._graph_ts_np = np.asarray(self.diag.ts, dtype=np.float64)
        self._graph_Q_np = np.asarray(self.diag.Qs, dtype=np.float64) * PARTICLE_NUMBER_SCALE
        self._graph_W_np = np.asarray(self.diag.Ws, dtype=np.float64) * PARTICLE_NUMBER_SCALE

    def _reset_pending_diag(self):
        self._pending_diag_dt = 0.0
        self._pending_impulse_y1 = 0.0
        self._pending_impulse_y2 = 0.0
        self._pending_impulse_x1 = 0.0
        self._pending_impulse_x2 = 0.0
        self._pending_dQ = 0.0
        self._pending_dW = 0.0

    def _accumulate_pending_diag(self, dt_phys: float):
        self._pending_diag_dt += dt_phys
        self._pending_impulse_y1 += self.state.impulse_piston_y1
        self._pending_impulse_y2 += self.state.impulse_piston_y2
        self._pending_impulse_x1 += self.state.impulse_piston_x1
        self._pending_impulse_x2 += self.state.impulse_piston_x2
        self._pending_dQ += self.state.dQ_step
        self._pending_dW += self.state.dW_by_step

    def _flush_pending_diag(self, hist_every: int):
        if self._pending_diag_dt <= 0.0:
            return
        saved = (
            self.state.impulse_piston_y1,
            self.state.impulse_piston_y2,
            self.state.impulse_piston_x1,
            self.state.impulse_piston_x2,
            self.state.dQ_step,
            self.state.dW_by_step,
        )
        self.state.impulse_piston_y1 = self._pending_impulse_y1
        self.state.impulse_piston_y2 = self._pending_impulse_y2
        self.state.impulse_piston_x1 = self._pending_impulse_x1
        self.state.impulse_piston_x2 = self._pending_impulse_x2
        self.state.dQ_step = self._pending_dQ
        self.state.dW_by_step = self._pending_dW
        self.diag.update(self.state, self._pending_diag_dt, hist_every=hist_every)
        (
            self.state.impulse_piston_y1,
            self.state.impulse_piston_y2,
            self.state.impulse_piston_x1,
            self.state.impulse_piston_x2,
            self.state.dQ_step,
            self.state.dW_by_step,
        ) = saved
        self._reset_pending_diag()

    def _sync_log_save_interval(self):
        self.diag.set_log_save_every(XLSX_SAVE_EVERY)

    def _clamp_playback_for_mode(self):
        max_scale = playback_max_for_unit()
        self.playback_scale = max(PLAYBACK_MIN, min(max_scale, self.playback_scale))
        self.playback_slider.value = self._slider_from_playback(self.playback_scale)
        self._refresh_playback_label()

    def _request_draw(self):
        self._force_draw = True

    def _should_draw(self) -> bool:
        if self._force_draw:
            self._force_draw = False
            return True
        intervals = self._effective_intervals()
        return (self._frame_id % intervals["draw_every"]) == 0

    def _playback_from_slider(self, t: float) -> float:
        t = max(0.0, min(1.0, t))
        log_max = playback_log_max_for_unit()
        scale = 10 ** (PLAYBACK_LOG_MIN + t * (log_max - PLAYBACK_LOG_MIN))
        return max(PLAYBACK_MIN, min(playback_max_for_unit(), scale))

    def _slider_from_playback(self, scale: float) -> float:
        log_max = playback_log_max_for_unit()
        scale = max(PLAYBACK_MIN, min(playback_max_for_unit(), scale))
        return (math.log10(scale) - PLAYBACK_LOG_MIN) / (log_max - PLAYBACK_LOG_MIN)

    def _refresh_playback_label(self):
        real_per_sim = 1.0 / max(PLAYBACK_MIN, self.playback_scale)
        self.playback_label_text = f"1 real s = {self.playback_scale:.1e} sim s"

    # -------------------------
    # State helpers
    # -------------------------
    def _init_state(self, N: int) -> SimState:
        x = np.random.uniform(SIM_X0 + 20, SIM_X1 - 20, size=N).astype(np.float32)
        y = np.random.uniform(SIM_Y0 + 20, PISTON_Y_INIT - 20, size=N).astype(np.float32)
        z = np.random.uniform(0.0 + 20, CYLINDER_DEPTH_PX - 20, size=N).astype(np.float32)

        bath_Tk = 273.16
        sigma = sigma_from_Tk(bath_Tk)
        vx = np.random.normal(0.0, sigma, size=N).astype(np.float32)
        vy = np.random.normal(0.0, sigma, size=N).astype(np.float32)
        vz = np.random.normal(0.0, sigma, size=N).astype(np.float32)

        return SimState(
            N=N, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
            piston_y=PISTON_Y_INIT,
            piston_u=U_INIT,
            thermal_on=False,
            bath_Tk=bath_Tk,
        )

    def reset_all(self, N: Optional[int] = None, reset_logs: bool = False):
        if N is None:
            N = self.state.N
        self.state = self._init_state(N)
        log_paths = make_timestamped_log_paths(get_app_dir()) if reset_logs else None
        self.diag.reset(reset_logs=reset_logs, log_paths=log_paths)
        self._prev_auto_T = None
        self._prev_auto_P = None
        self._prev_auto_V = None
        self._graph_ts_np = None
        self._graph_Q_np = None
        self._graph_W_np = None

    def _begin_particle_prompt(self, reason: str, resume_running: Optional[bool] = None):
        if resume_running is None:
            resume_running = self.running
        self._prompt_active = True
        self._prompt_text = ""
        self._prompt_error = ""
        self._prompt_reason = reason
        self._running_before_prompt = resume_running
        self.running = False
        self._set_input_focus(None)
        self._request_draw()

    def _finish_particle_prompt(self):
        self._prompt_active = False
        self._prompt_error = ""
        self.running = self._running_before_prompt
        self._request_draw()

    def _apply_particle_settings(self, n_sim: int, unit_mode: str):
        _apply_particle_settings(n_sim, unit_mode)
        self.diag.pt_warmup_phys = _compute_pt_warmup_phys(DEFAULT_N)
        self._prompt_mode = "mols" if N_UNIT == "mols" else "particles"
        self._clamp_playback_for_mode()
        self._sync_log_save_interval()
        self._request_draw()

    def _parse_particle_prompt(self, raw: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        if self._prompt_mode == "mols":
            try:
                val = float(raw)
            except ValueError:
                return None, None, "mols는 숫자로 입력하세요 (예: 1.0)"
            if not (0.0 < val <= 2.0):
                return None, None, "mols 범위: 0~2"
            n_sim = int(val * AVOGADRO_NUMBER_SIMULATOR)
            if n_sim < 1:
                return None, None, "mols 값이 너무 작습니다"
            return n_sim, "mols", None

        if not raw.isdigit():
            return None, None, "particles는 정수로 입력하세요"
        n_sim = int(raw)
        if n_sim < 1 or n_sim > AVOGADRO_NUMBER_SIMULATOR:
            return None, None, f"particles 범위: 1~{AVOGADRO_NUMBER_SIMULATOR}"
        return n_sim, "particles", None

    def _submit_particle_prompt(self, canceled: bool):
        raw = self._prompt_text.strip()
        if canceled or raw == "":
            if self._prompt_reason == "reset":
                self.reset_all(reset_logs=True)
            self._finish_particle_prompt()
            return

        n_sim, unit_mode, err = self._parse_particle_prompt(raw)
        if err:
            self._prompt_error = err
            return
        self._apply_particle_settings(n_sim, unit_mode)
        reset_logs = self._prompt_reason == "reset"
        self.reset_all(n_sim, reset_logs=reset_logs)
        self._finish_particle_prompt()

    def _estimate_TPV_now(self, dt_phys: float) -> Tuple[float, float, float]:
        """Instantaneous estimates for auto-pause (per-frame), independent of Diagnostics windowing.

        Returns:
            T_now_K, P_now_kPa, V_now_L
        """
        # Temperature from kinetic energy (3D)
        v2_m = _compute_v2_m(self.state.vx, self.state.vy, SPEED_SCALE, vz=self.state.vz)
        T_now = _compute_temperature_from_v2(v2_m, MASS, dof=3)

        # Volume from geometry
        V_now = _compute_volume_L(self.state.piston_y, clamp_height=True)

        # Pressure estimate from this frame's wall impulses
        dt_phys = max(1e-12, dt_phys)
        area_tb, area_lr = _compute_area_tb_lr(self.state.piston_y)
        imp_per_area_top = abs(self.state.impulse_piston_y2) / area_tb
        imp_per_area_bottom = abs(self.state.impulse_piston_y1) / area_tb
        imp_per_area_left = abs(self.state.impulse_piston_x1) / area_lr
        imp_per_area_right = abs(self.state.impulse_piston_x2) / area_lr
        imp_per_area_avg = (imp_per_area_top + imp_per_area_bottom + imp_per_area_left + imp_per_area_right) / 4.0
        P_now_pa = abs(imp_per_area_avg) / dt_phys
        P_now_kpa = P_now_pa / ADJUSTED_KPA_PA

        return T_now, P_now_kpa, V_now

    def _check_auto_pause(self):
        # Crossing detection: pause when we pass the target in either direction.
        T_now, P_now, V_now = self._estimate_TPV_now(getattr(self, "_last_dt_phys", 0.0))

        # initialize history
        if self._prev_auto_T is None:
            self._prev_auto_T = T_now
            self._prev_auto_P = P_now
            self._prev_auto_V = V_now
            return

        T_prev = self._prev_auto_T
        P_prev = self._prev_auto_P
        V_prev = self._prev_auto_V

        hit = False
        if self.target_T >= 0.0:
            hit = hit or ((T_prev - self.target_T) * (T_now - self.target_T) <= 0.0)
        if self.target_P >= 0.0:
            hit = hit or ((P_prev - self.target_P) * (P_now - self.target_P) <= 0.0)
        if self.target_V >= 0.0:
            hit = hit or ((V_prev - self.target_V) * (V_now - self.target_V) <= 0.0)

        if hit:
            self.running = False
        # update history for next frame
        self._prev_auto_T = T_now
        self._prev_auto_P = P_now
        self._prev_auto_V = V_now

    def _apply_berendsen(self, dt_phys: float):
        if not self.state.thermal_on:
            return
        if BERENDSEN_EVERY <= 0 or (self._frame_id % BERENDSEN_EVERY) != 0:
            return
        if dt_phys <= 0.0 or BERENDSEN_TAU <= 0.0:
            return

        alpha = min(BERENDSEN_MAX_ALPHA, dt_phys / BERENDSEN_TAU)
        if alpha <= 0.0:
            return

        vx = self.state.vx
        vy = self.state.vy
        vz = self.state.vz
        if vx.size == 0:
            return

        vxm = float(np.mean(vx))
        vym = float(np.mean(vy))
        vzm = float(np.mean(vz))
        ux = vx - vxm
        uy = vy - vym
        uz = vz - vzm

        u2_m = _compute_v2_m(ux, uy, SPEED_SCALE, vz=uz)
        # T_curr and E0 both reduce u2_m; compute the sum once and derive both.
        sum_u2_m = float(np.sum(u2_m))
        N = u2_m.size
        T_curr = float(MASS_F32 * (sum_u2_m / N) / (3.0 * KB)) if N > 0 else 0.0
        if T_curr <= 0.0:
            return

        T_target = float(self.state.bath_Tk)
        lam2 = 1.0 + alpha * (T_target / T_curr - 1.0)
        if not np.isfinite(lam2) or lam2 <= 0.0:
            return
        lam = math.sqrt(lam2)
        if not np.isfinite(lam) or lam <= 0.0:
            return

        E0 = 0.5 * MASS_F32 * sum_u2_m
        if not np.isfinite(E0):
            E0 = 0.0

        self.state.vx[:] = (vxm + lam * ux).astype(vx.dtype, copy=False)
        self.state.vy[:] = (vym + lam * uy).astype(vy.dtype, copy=False)
        self.state.vz[:] = (vzm + lam * uz).astype(vz.dtype, copy=False)

        deltaE = (lam2 - 1.0) * E0
        self.state.dQ_step += float(deltaE)

    # -------------------------
    # Simulation update
    # -------------------------
    def on_update(self, dt: float):
        # Keep targets in sync even if on_change isn't firing
        self._sync_targets_from_inputs()
        self.playback_scale = self._playback_from_slider(self.playback_slider.value)
        self._clamp_playback_for_mode()
        self._refresh_playback_label()

        intervals = self._effective_intervals()
        is_fast = is_fast_playback(self.playback_scale)
        if self._fast_mode_active and not is_fast:
            self._flush_pending_diag(intervals["hist_every"])
        if self._fast_mode_active != is_fast:
            if is_fast:
                self._reset_pending_diag()
            self._sync_log_save_interval()
        self._fast_mode_active = is_fast

        if not self.running:
            return
        dt_phys = dt * self.playback_scale
        self._last_dt_phys = dt_phys
        self._frame_id += 1

        # adaptive substeps: bound by time-to-wall for approaching particles (less over-splitting)
        state = self.state
        vx = state.vx
        vy = state.vy
        px = state.piston_u

        width = (SIM_X1 - SIM_X0)
        height = (state.piston_y - SIM_Y0)

        # min(c / x_i) == c / max(x_i); compute via scalar reductions to avoid N-sized intermediates.
        times = []
        if vx.size > 0:
            vx_max = float(vx.max())
            vx_min = float(vx.min())
            max_abs_vx = vx_max if vx_max > -vx_min else -vx_min
            if max_abs_vx > 0.0:
                times.append(width / max(max_abs_vx, 1e-12))
        if vy.size > 0:
            vy_max = float(vy.max())
            vy_min = float(vy.min())
            if vy_min < 0.0:
                times.append(height / max(-vy_min, 1e-12))
            if vy_max > px:
                times.append(height / max(vy_max - px, 1e-12))

        if times:
            dt_cap = min(times)
        else:
            dt_cap = dt_phys

        n_substeps = int(math.ceil(dt_phys / dt_cap))
        n_substeps = max(n_substeps, 1)
        n_substeps = min(n_substeps, MAX_SUBSTEPS)
        sub_dt = dt_phys / n_substeps
        self._last_substeps = n_substeps

        impulse_y1 = 0.0
        impulse_y2 = 0.0
        impulse_x1 = 0.0
        impulse_x2 = 0.0
        dQ = 0.0
        dW = 0.0
        engine_step = self.engine.step
        for _ in range(n_substeps):
            engine_step(state, sub_dt)
            impulse_y1 += state.impulse_piston_y1
            impulse_y2 += state.impulse_piston_y2
            impulse_x1 += state.impulse_piston_x1
            impulse_x2 += state.impulse_piston_x2
            dQ += state.dQ_step
            dW += state.dW_by_step

        # combine step accumulators for diagnostics
        state.impulse_piston_y1 = impulse_y1
        state.impulse_piston_y2 = impulse_y2
        state.impulse_piston_x1 = impulse_x1
        state.impulse_piston_x2 = impulse_x2
        state.dQ_step = dQ
        state.dW_by_step = dW

        # BGK-like rare collisions once per frame (scaled by dt_phys)
        # Use collision_rate from STP rescaled by particle number ratio
        # collision_rate = 1e10 * (DEFAULT_N / AVOGADRO_NUMBER)  # strict scaling ~1e-9 Hz
        # collision_rate = 1e10 * (DEFAULT_N / AVOGADRO_NUMBER_SIMULATOR) # user suggestion ~1e10 Hz (very strong)
        
        # Using the user's specific formula:
        collision_rate = COLLISION_RATE_SCALING * 1 * (DEFAULT_N / AVOGADRO_NUMBER_SIMULATOR)
        self._bgk_mix(dt_phys, collision_rate)

        # Gentle Berendsen rescaling on peculiar velocities (visual smoothing)
        self._apply_berendsen(dt_phys)

        if is_fast:
            self._accumulate_pending_diag(dt_phys)
            if (self._frame_id % intervals["diag_every"]) == 0:
                self._flush_pending_diag(intervals["hist_every"])
        else:
            self.diag.update(self.state, dt_phys, hist_every=intervals["hist_every"])

        self._maybe_refresh_graph_cache()
        self._check_auto_pause()

    # -------------------------
    # BGK mixing
    # -------------------------
    def _bgk_mix(self, dt_phys: float, collision_rate: float):
        if collision_rate <= 0.0:
            return

        expected = self.state.N * collision_rate * dt_phys
        # BGK mixing acts on particle pairs; "expected" counts single particles.
        # So, the expected number of *pairs* is expected/2.
        n_pairs = int(max(0, min(self.state.N // 2, expected / 2)))
        if n_pairs <= 0:
            return
        preserve_energy = self.state.thermal_on and BGK_PRESERVE_ENERGY
        E_pre = 0.0
        if preserve_energy:
            vx_all = self.state.vx
            vy_all = self.state.vy
            vz_all = self.state.vz
            vxm_all = float(np.mean(vx_all))
            vym_all = float(np.mean(vy_all))
            vzm_all = float(np.mean(vz_all))
            ux_all = vx_all - vxm_all
            uy_all = vy_all - vym_all
            uz_all = vz_all - vzm_all
            u2_m_pre = _compute_v2_m(ux_all, uy_all, SPEED_SCALE, vz=uz_all)
            E_pre = 0.5 * MASS_F32 * float(np.sum(u2_m_pre))
            if not np.isfinite(E_pre) or E_pre <= 0.0:
                preserve_energy = False
        # pick disjoint pairs when possible to avoid double-hits in one mixer call
        if 2 * n_pairs <= self.state.N:
            perm = np.random.permutation(self.state.N)[: 2 * n_pairs]
            a_idx = perm[:n_pairs]
            b_idx = perm[n_pairs:]
        else:
            a_idx = np.random.randint(0, self.state.N, size=n_pairs, dtype=int)
            b_idx = np.random.randint(0, self.state.N, size=n_pairs, dtype=int)
            same = (a_idx == b_idx)
            if np.any(same):
                b_idx[same] = (b_idx[same] + 1) % self.state.N

        vx_a = self.state.vx[a_idx]
        vy_a = self.state.vy[a_idx]
        vz_a = self.state.vz[a_idx]
        vx_b = self.state.vx[b_idx]
        vy_b = self.state.vy[b_idx]
        vz_b = self.state.vz[b_idx]

        vx_cm = 0.5 * (vx_a + vx_b)
        vy_cm = 0.5 * (vy_a + vy_b)
        vz_cm = 0.5 * (vz_a + vz_b)

        dvx = vx_a - vx_b
        dvy = vy_a - vy_b
        dvz = vz_a - vz_b
        dv2_old = dvx * dvx + dvy * dvy + dvz * dvz

        if self.state.thermal_on:
            # Lowe–Andersen: resample relative velocity from Maxwellian at bath_Tk
            # In large-N mode, BGK only reshapes the distribution; energy is set by Berendsen.
            Tk = max(1e-6, float(self.state.bath_Tk))
            sigma_g = math.sqrt(2.0 * KB * Tk / MASS) / SPEED_SCALE
            dtype = self.state.vx.dtype
            g_x = np.random.normal(0.0, sigma_g, size=n_pairs).astype(dtype)
            g_y = np.random.normal(0.0, sigma_g, size=n_pairs).astype(dtype)
            g_z = np.random.normal(0.0, sigma_g, size=n_pairs).astype(dtype)
            dv2_new = g_x * g_x + g_y * g_y + g_z * g_z

            self.state.vx[a_idx] = vx_cm + 0.5 * g_x
            self.state.vy[a_idx] = vy_cm + 0.5 * g_y
            self.state.vz[a_idx] = vz_cm + 0.5 * g_z
            self.state.vx[b_idx] = vx_cm - 0.5 * g_x
            self.state.vy[b_idx] = vy_cm - 0.5 * g_y
            self.state.vz[b_idx] = vz_cm - 0.5 * g_z

            if not preserve_energy:
                delta_e = 0.25 * MASS_F32 * (dv2_new - dv2_old) * (SPEED_SCALE_F32 * SPEED_SCALE_F32)
                self.state.dQ_step += float(np.sum(delta_e))
        else:
            # Elastic mixing: randomize relative-velocity direction to couple 3D DOF
            dv_mag = np.sqrt(dv2_old)
            g_x = np.random.normal(0.0, 1.0, size=n_pairs)
            g_y = np.random.normal(0.0, 1.0, size=n_pairs)
            g_z = np.random.normal(0.0, 1.0, size=n_pairs)
            g_norm = np.sqrt(g_x * g_x + g_y * g_y + g_z * g_z)
            g_norm = np.where(g_norm > 0.0, g_norm, 1.0)
            g_x = (dv_mag * g_x / g_norm).astype(self.state.vx.dtype)
            g_y = (dv_mag * g_y / g_norm).astype(self.state.vy.dtype)
            g_z = (dv_mag * g_z / g_norm).astype(self.state.vz.dtype)

            self.state.vx[a_idx] = vx_cm + 0.5 * g_x
            self.state.vy[a_idx] = vy_cm + 0.5 * g_y
            self.state.vz[a_idx] = vz_cm + 0.5 * g_z
            self.state.vx[b_idx] = vx_cm - 0.5 * g_x
            self.state.vy[b_idx] = vy_cm - 0.5 * g_y
            self.state.vz[b_idx] = vz_cm - 0.5 * g_z

        if preserve_energy:
            vx_all = self.state.vx
            vy_all = self.state.vy
            vz_all = self.state.vz
            vxm_all = float(np.mean(vx_all))
            vym_all = float(np.mean(vy_all))
            vzm_all = float(np.mean(vz_all))
            ux_all = vx_all - vxm_all
            uy_all = vy_all - vym_all
            uz_all = vz_all - vzm_all
            u2_m_post = _compute_v2_m(ux_all, uy_all, SPEED_SCALE, vz=uz_all)
            E_post = 0.5 * MASS_F32 * float(np.sum(u2_m_post))
            if np.isfinite(E_post) and E_post > 0.0:
                lam2 = E_pre / E_post
                if np.isfinite(lam2) and lam2 > 0.0:
                    lam = math.sqrt(lam2)
                    self.state.vx[:] = (vxm_all + lam * ux_all).astype(vx_all.dtype, copy=False)
                    self.state.vy[:] = (vym_all + lam * uy_all).astype(vy_all.dtype, copy=False)
                    self.state.vz[:] = (vzm_all + lam * uz_all).astype(vz_all.dtype, copy=False)
            else:
                delta_e = 0.25 * MASS_F32 * (dv2_new - dv2_old) * (SPEED_SCALE_F32 * SPEED_SCALE_F32)
                self.state.dQ_step += float(np.sum(delta_e))

    # -------------------------
    # Drawing helpers
    # -------------------------
    def _draw_text_cached(
        self,
        key,
        text,
        x,
        y,
        color,
        font_size=12,
        width=None,
        align="left",
        anchor_x="left",
        anchor_y="baseline",
        multiline=False,
    ):
        text_value = "" if text is None else str(text)
        color_value = tuple(color)
        style = (font_size, width, align, anchor_x, anchor_y, multiline)
        if multiline or ("\n" in text_value):
            lines = text_value.splitlines()
            font = self._get_font(font_size)
            line_h = font.get_linesize()
            for i, line in enumerate(lines):
                y_line = y - i * line_h
                self._draw_text_cached(
                    (key, i),
                    line,
                    x,
                    y_line,
                    color,
                    font_size=font_size,
                    width=width,
                    align=align,
                    anchor_x=anchor_x,
                    anchor_y=anchor_y,
                    multiline=False,
                )
            return

        entry = self._text_cache.get(key)
        if (
            entry is None
            or entry["style"] != style
            or entry["text"] != text_value
            or entry["color"] != color_value
        ):
            font = self._get_font(font_size)
            surface = font.render(text_value, True, color_value[:3])
            entry = {"style": style, "text": text_value, "color": color_value, "surface": surface}
            self._text_cache[key] = entry
        else:
            surface = entry["surface"]

        rect = surface.get_rect()
        if anchor_x == "center":
            draw_x = x - rect.width * 0.5
        elif anchor_x == "right":
            draw_x = x - rect.width
        else: # left
            draw_x = x

        if anchor_y == "top":
            draw_y = y
        elif anchor_y == "center":
            draw_y = y + rect.height * 0.5
        elif anchor_y == "bottom":
            draw_y = y + rect.height
        else: # baseline
            ascent = self._get_font(font_size).get_ascent()
            draw_y = y + ascent

        self.screen.blit(surface, (draw_x, _to_screen_y(draw_y)))

    def draw_sim(self):
        # Targets are already synced in on_update each frame
        # cylinder outline
        draw_lrbt_rectangle_outline(self.screen, SIM_X0, SIM_X1, SIM_Y0, self.state.piston_y, Colors.BLACK, 2)
        # piston line
        draw_line(self.screen, SIM_X0, self.state.piston_y, SIM_X1, self.state.piston_y, Colors.BLACK, 3)

        # thermal walls visual: red zigzag on L/R/B when coupled to bath
        if self.state.thermal_on:
            def zigzag(x0, y0, x1, y1, amp=6, period=14):
                pts = []
                if x0 == x1:  # vertical
                    length = abs(y1 - y0)
                    n = max(1, int(length // period))
                    direction = 1
                    y = y0
                    dy = (y1 - y0) / n
                    for i in range(n + 1):
                        pts.append((x0 + direction * amp, y))
                        direction *= -1
                        y += dy
                elif y0 == y1:  # horizontal
                    length = abs(x1 - x0)
                    n = max(1, int(length // period))
                    direction = 1
                    x = x0
                    dx = (x1 - x0) / n
                    for i in range(n + 1):
                        pts.append((x, y0 + direction * amp))
                        direction *= -1
                        x += dx
                if len(pts) >= 2:
                    draw_line_strip(self.screen, pts, Colors.RED, 2)

            zigzag(SIM_X0, SIM_Y0, SIM_X0, self.state.piston_y)   # left
            zigzag(SIM_X1, SIM_Y0, SIM_X1, self.state.piston_y)   # right
            zigzag(SIM_X0, SIM_Y0, SIM_X1, SIM_Y0)                # bottom

        # particles
        n_draw = min(self.state.N, MAX_RENDER_PARTICLES)
        if self.state.N <= MAX_RENDER_PARTICLES:
            idx_iter = range(self.state.N)
        else:
            step = max(1, self.state.N // MAX_RENDER_PARTICLES)
            idx_iter = range(0, self.state.N, step)
        depth_inv = 1.0 / max(1e-9, CYLINDER_DEPTH_PX)
        radius_span = RADIUS_MAX - RADIUS_MIN
        drawn = 0
        for i in idx_iter:
            z_norm = self.state.z[i] * depth_inv
            if z_norm < 0.0:
                z_norm = 0.0
            elif z_norm > 1.0:
                z_norm = 1.0
            radius = RADIUS_MIN + radius_span * z_norm
            draw_circle_filled(self.screen, self.state.x[i], self.state.y[i], radius, Colors.BLUE)
            drawn += 1
            if drawn >= n_draw:
                break

        # labels moved to info panel

    def _draw_panel_frame(self, rect, title: str):
        x0, x1, y0, y1 = rect
        draw_lrbt_rectangle_outline(self.screen, x0, x1, y0, y1, Colors.GRAY, 1)
        if title:
            self._draw_text_cached(key=("panel_title", rect), text=title, x=x0 + 5, y=y1-5, color=Colors.BLACK, font_size=12, anchor_y="top")
            # arcade.draw_line(x0, y1 - PANEL_TITLE_H, x1, y1 - PANEL_TITLE_H, arcade.color.LIGHT_GRAY, 1)

    def _map_to_rect(self, rect, xs, ys, xlim, ylim):
        x0, x1, y0, y1 = rect
        xmin, xmax = xlim
        ymin, ymax = ylim
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        xs = np.clip(xs, xmin, xmax)
        ys = np.clip(ys, ymin, ymax)
        X = x0 + (xs - xmin) / max(1e-12, (xmax - xmin)) * (x1 - x0)
        Y = y0 + (ys - ymin) / max(1e-12, (ymax - ymin)) * (y1 - y0)
        return X, Y

    def draw_hist(self):
        rect = (G1[0], G1[1], G1[2], G1[3])
        self._draw_panel_frame(rect, "N(v) vs v (m/s)")

        counts = self.diag.hist_counts
        if counts.max() <= 0:
            return
        # bars
        x0, x1, y0, y1 = rect
        pad = 30
        bx0, bx1 = x0 +pad/2, x1 -pad/2 
        by0, by1 = y0 + pad, y1 - pad

        nb = len(counts)
        w = (bx1 - bx0) / nb
        cmax = counts.max()

        for i, c in enumerate(counts):
            h = (c / cmax) * (by1 - by0)
            xl = bx0 + i * w
            xr = xl + w * 0.9
            draw_lrbt_rectangle_filled(self.screen, xl, xr, by0, by0 + h, Colors.LIGHT_GRAY)

        # Maxwell speed PDF (3D): use bath_Tk when coupled, otherwise current kinetic T
        v_bins = self.diag.hist_vmax
        dv = v_bins / nb
        v_centers = (np.arange(nb) + 0.5) * dv


        # x-axis ticks for speed intuition
        tick_vals = np.linspace(0, v_bins, 5)
        for i, tv in enumerate(tick_vals):
            tx = bx0 + (tv / v_bins) * (bx1 - bx0)
            # draw_line(self.screen, tx, y0 + 4, tx, y0 + 12, Colors.GRAY, 1)
            self._draw_text_cached(
                ("hist_tick", i),
                f"{tv:.0f}",
                tx - 10,
                by0,
                Colors.GRAY,
                9,
                anchor_y="top",
            )

        # self._draw_text_cached(
        #     "hist_vmax",
        #     f"v_max={self.diag.hist_vmax:.0f} m/s",
        #     x1 - 150,
        #     y0 + 6,
        #     Colors.BLACK,
        #     10,
        #     anchor_y="bottom",
        # )
                # current kinetic T from state
        v2_m = _compute_v2_m(self.state.vx, self.state.vy, SPEED_SCALE, vz=self.state.vz)
        T_curr = _compute_temperature_from_v2(v2_m, MASS, dof=3)
        T_pdf = self.state.bath_Tk 
        tiks_line_h = self._get_font(9).get_linesize()

        if T_pdf > 1e-12:
            pdf = (v_centers * v_centers) * np.exp(
                -MASS * v_centers * v_centers / (2.0 * KB * T_pdf)
            )
            pdf_norm = pdf / max(1e-30, pdf.max())
            X_pdf = bx0 + (v_centers / v_bins) * (bx1 - bx0)
            Y_pdf = by0 + pdf_norm * (by1 - by0)
            draw_line_strip(self.screen, list(zip(X_pdf, Y_pdf)), Colors.BLUE, 2)
            label_T = T_pdf #if self.state.thermal_on else T_curr
            self._draw_text_cached(
                "hist_temp_label",
                f"T_bath~{label_T:.2f} K / T_gas~{T_curr:.2f} K",
                x0 + 10,
                y0 + tiks_line_h,
                Colors.BLUE,
                10,
                anchor_y="top",
            )

    def draw_timeseries(
        self, rect, title, t, y, ylabel=None, y_fixed=None, y_ref=None
    ):
        self._draw_panel_frame(rect, title)
        if t.size < 2:
            return

        # window last points
        tmin, tmax = t.min(), t.max()
        if tmax - tmin < 1e-9:
            tmin, tmax = 0.0, 1.0

        # y-lim with margin or fixed
        if y_fixed is not None:
            ymin, ymax = y_fixed
        else:
            ymin, ymax = float(np.min(y)), float(np.max(y))
            if abs(ymax - ymin) < 1e-9:
                ymin -= 1.0
                ymax += 1.0
            else:
                m = 0.08 * (ymax - ymin)
                ymin -= m
                ymax += m

        x0, x1, y0, y1 = rect
        if y_ref is not None and ymin <= y_ref <= ymax:
            y_span = max(1e-12, ymax - ymin)
            y_line = y0 + (y_ref - ymin) / y_span * (y1 - y0)
            draw_dotted_line_horizontal_lrbt(
                self.screen,
                x0 + 1,
                x1 - 1,
                y_line,
                (180, 180, 180),
                width=1,
                dash=5,
                gap=4,
            )

        X, Y = self._map_to_rect(rect, t, y, (tmin, tmax), (ymin, ymax))
        draw_line_strip(self.screen, list(zip(X, Y)), Colors.BLACK, 2)

        if ylabel is not None:
            x0, x1, y0, y1 = rect
            self._draw_text_cached(
                ("timeseries_label", rect),
                f"{ylabel}: {y[-1]:.3g}",
                x0 + 8,
                y0 + 6,
                Colors.BLACK,
                10,
                # anchor_y="top",
            )

    def draw_PT(self):
        rect = (G4[0], G4[1], G4[2], G4[3])
        self._draw_panel_frame(rect, "P-T trajectory (P in kPa, T in K)")

        if len(self.diag.PT) < 2:
            return
        PT_arr = np.asarray(self.diag.PT, dtype=np.float64)
        P = PT_arr[:, 0] / ADJUSTED_KPA_PA
        T = PT_arr[:, 1]


        Pmin, Pmax = 0.0, float(np.max(P)) * 1.5
        Tmin, Tmax = 100., 500.,

        # add margins
        Pm = 0.5 * (Pmax - Pmin)
        Tm = 0.5 * (Tmax - Tmin)
        Pmin -= Pm
        Pmax += Pm
        Tmin -= Tm
        Tmax += Tm

        X, Y = self._map_to_rect(rect, T, P, (Tmin, Tmax), (Pmin, Pmax))  # x=T, y=P
        draw_line_strip(self.screen, list(zip(X, Y)), Colors.BLACK, 2)

        x0, x1, y0, y1 = rect
        self._draw_text_cached(
            "pt_label",
            f"T={T[-1]:.2f} K, P={P[-1]:.2f} kPa",
            x0 + 8,
            y0 + 6,
            Colors.BLACK,
            10,
        )

    def draw_table(self, rect=None):
        if rect is None:
            rect = (G6[0], G6[1], G6[2], G6[3])
        self._draw_panel_frame(rect, "Table")
        if len(self.diag.ts) == 0:
            return

        x0, x1, y0, y1 = rect
        pad_x = 8
        header_y = y1 - PANEL_TITLE_H - 6
        header_font_size = 10
        row_font_size = 10
        header_line_h = self._get_font(header_font_size).get_linesize()
        row_h = self._get_font(row_font_size).get_linesize() + 2
        total_w = (x1 - x0) - 2 * pad_x
        if N_UNIT == "particles":
            headers = ["time (s)", "vx_rms (m/s)", "vy_rms (m/s)", "vz_rms (m/s)", "f_top (N)"]
        else:
            headers = ["time (s)", "P (kPa)", "V (L)", "T (K)"]
        min_col_w = 80
        max_cols = max(2, min(len(headers), int(total_w / min_col_w)))
        headers = headers[:max_cols]
        col_w = total_w / len(headers)
        col_x = [x0 + pad_x + i * col_w for i in range(len(headers))]
        for i, h in enumerate(headers):
            self._draw_text_cached(("table_header", i), h, col_x[i], header_y, Colors.BLACK, header_font_size)

        max_rows = int((header_y - y0 - 6) / max(1, row_h))
        start = max(0, len(self.diag.ts) - max_rows)
        for r, idx in enumerate(range(start, len(self.diag.ts))):
            y = header_y - (header_line_h + 4) - r * row_h
            t = self.diag.ts[idx]
            if N_UNIT == "particles":
                vx_rms = self.diag.vx_rms_s[idx]
                vy_rms = self.diag.vy_rms_s[idx]
                vz_rms = self.diag.vz_rms_s[idx]
                f_top = self.diag.f_top_s[idx]
                values = [
                    f"{t:.3g}",
                    f"{vx_rms:.3g}",
                    f"{vy_rms:.3g}",
                    f"{vz_rms:.3g}",
                    f"{f_top:.3g}",
                ]
            else:
                p_kpa = self.diag.Ps[idx] / ADJUSTED_KPA_PA
                T = self.diag.Ts[idx]
                V = self.diag.Vs[idx]
                values = [
                    f"{t:.3g}",
                    f"{p_kpa:.2f}",
                    f"{V:.4g}",
                    f"{T:.2f}",
                ]
            values = values[:max_cols]
            for i, v in enumerate(values):
                self._draw_text_cached(("table_cell", r, i), v, col_x[i], y, Colors.BLACK, row_font_size)

    def draw_info_panel(self, rect):
        self._draw_panel_frame(rect, "Info")
        x0, x1, y0, y1 = rect
        pad_x = 8
        pad_y = 8
        font_size = 10
        line_h = self._get_font(font_size).get_linesize() + 2

        mode = "열접촉" if self.state.thermal_on else "단열"
        u_m_s = self.state.piston_u * SPEED_SCALE
        height_m = (self.state.piston_y - SIM_Y0) * PX_TO_M
        width_m = (SIM_X1 - SIM_X0) * PX_TO_M
        depth_m = CYLINDER_DEPTH_PX * PX_TO_M

        volume_L = _compute_volume_L(self.state.piston_y, clamp_height=True)
        if N_UNIT == "mols":
            n_line = f"n: {self.state.N / AVOGADRO_NUMBER_SIMULATOR:.3g} mol"
        else:
            n_line = f"N: {self.state.N} particles"

        lines = [
            f"T_bath: {self.state.bath_Tk:.2f} K",
            f"walls: {mode}",
            n_line,
            f"piston u: {u_m_s * 100:+.2f} cm/s",
            f"(H, W, D): ({height_m:.2f},{width_m:.2f},{depth_m:.2f}) m",
            f"volume: {volume_L:.2f} L",
            f"t: {self.diag.t:.4f} s",
            # f"substeps: {getattr(self, '_last_substeps', self._substeps)}",
            "auto pause at:",
            f"{self.target_T:.2f}K" if self.target_T >0 else "",
            f", {self.target_P:.2f}kPa" if self.target_P >0 else "",
            f", {self.target_V:.3g}L" if self.target_V >0 else "",
        ]

        y = y1 - PANEL_TITLE_H - 6
        for i, line in enumerate(lines):
            if y < y0 + pad_y:
                break
            self._draw_text_cached(("info_line", i), line, x0 + pad_x, y, Colors.BLACK, font_size)
            y -= line_h

    def draw_particle_prompt(self):
        if not self._prompt_active:
            return
        draw_lrbt_rectangle_filled(self.screen, 0, SCREEN_W, 0, SCREEN_H, (0, 0, 0, 150))

        panel_w = 560
        panel_h = 240
        cx = SCREEN_W / 2
        cy = SCREEN_H / 2
        x0 = cx - panel_w / 2
        x1 = cx + panel_w / 2
        y0 = cy - panel_h / 2
        y1 = cy + panel_h / 2
        draw_lrbt_rectangle_filled(self.screen, x0, x1, y0, y1, Colors.WHITE)
        draw_lrbt_rectangle_outline(self.screen, x0, x1, y0, y1, Colors.GRAY, 2)

        title = "입자 수 설정"
        mode_label = "mols" if self._prompt_mode == "mols" else "particles"
        if N_UNIT == "mols":
            current_line = f"기본 값: {self.state.N / AVOGADRO_NUMBER_SIMULATOR:.3g} mol"
        else:
            current_line = f"기본 값 N: {self.state.N} particles"
        input_line = f"숫자 입력: {self._prompt_text}"
        hints = [
            input_line,
            f"단위: {mode_label}  ([M] key = mol, [P] key = particles)", "\n",
            "참고: "
            f"범위: mols 0~2 / particles 1~{AVOGADRO_NUMBER_SIMULATOR}",
            current_line,
            "Enter=적용, ESC=기본값으로 진행, Backspace=삭제",
        ]

        title_size = 14
        hint_size = 11
        self._draw_text_cached("prompt_title", title, x0 + 16, y1 - 28, Colors.BLACK, title_size)
        line_h = self._get_font(hint_size).get_linesize() + 6
        y = y1 - 56
        for i, line in enumerate(hints):
            self._draw_text_cached(("prompt_hint", i), line, x0 + 16, y, Colors.BLACK, hint_size)
            y -= line_h

        if self._prompt_error:
            self._draw_text_cached(
                "prompt_error",
                self._prompt_error,
                x0 + 16,
                y0 + 18,
                Colors.RED,
                hint_size,
            )

    def _is_widget_focused(self, widget) -> bool:
        return bool(getattr(widget, "focused", False))

    def _draw_input_box(self, key: str, box: InputBox):
        x0 = box.x
        x1 = box.x + box.width
        y0 = box.y
        y1 = box.y + box.height
        draw_lrbt_rectangle_filled(self.screen, x0, x1, y0, y1, Colors.WHITE)
        if self._is_widget_focused(box):
            color = Colors.BLUE
            width = 2
        else:
            color = Colors.DARK_GRAY
            width = 1
        draw_lrbt_rectangle_outline(self.screen, x0, x1, y0, y1, color, width)
        self._draw_text_cached(
            ("input_text", key),
            box.text,
            x0 + 6,
            y0, #+ box.height * 0.5,
            Colors.BLACK,
            box.font_size,
            anchor_y="bottom",
        )

    def draw_controls_ui(self):
        self._draw_text_cached(
            "playback_title",
            self._playback_title_text,
            self._playback_title_pos[0],
            self._playback_title_pos[1],
            Colors.BLACK,
            11,
            anchor_y="top",
        )
        self.playback_slider.draw(self.screen)
        self._draw_text_cached(
            "playback_label",
            self.playback_label_text,
            self._playback_label_pos[0],
            self._playback_label_pos[1],
            Colors.BLACK,
            11,
            anchor_y="top",
        )

        self._draw_text_cached(
            "target_title",
            self._target_title_text,
            self._target_title_pos[0],
            self._target_title_pos[1],
            Colors.BLACK,
            11,
            anchor_y="top",
        )
        for key, label in self._target_row_labels.items():
            pos = self._target_row_positions[key]
            self._draw_text_cached(
                ("target_label", key),
                label,
                pos[0],
                pos[1],
                Colors.BLACK,
                11,
                anchor_y="center",
            )
            self._draw_input_box(key, self.target_inputs[key])

    def on_draw(self):
        self.screen.fill(Colors.WHITE)
        self.draw_sim()

        if self._graph_ts_np is None:
            self._maybe_refresh_graph_cache(force=True)
        ts_np = self._graph_ts_np if self._graph_ts_np is not None else np.asarray([], dtype=np.float64)
        Q_np = self._graph_Q_np if self._graph_Q_np is not None else np.asarray([], dtype=np.float64)
        W_np = self._graph_W_np if self._graph_W_np is not None else np.asarray([], dtype=np.float64)

        self.draw_hist()
        self.draw_timeseries(
            (G2[0], G2[1], G2[2], G2[3]),
            title="Q(t): Total heat absorbed from bath [J]",
            t=ts_np,
            y=Q_np,
            ylabel="Q (J)",
            y_ref=0.0,
        )
        self.draw_timeseries(
            (G3[0], G3[1], G3[2], G3[3]),
            "W(t): Total work done by the gas [J]",
            ts_np,
            W_np,
            ylabel="W (J)",
            y_ref=0.0,
        )
        self.draw_table(BOTTOM_TABLE)
        self.draw_info_panel(BOTTOM_INFO)
        self._draw_panel_frame(BOTTOM_CONTROLS, "Controls")
        self.draw_controls_ui()

        # controls
        help_text = (
            "Controls:  SPACE=Play/Pause | R=Reset"
            "           A=Toggle thermal/adiabatic"
            "           Left/Right= bath T (K)   |  Q:250K  W:273.16K  E:350K"
            f"          Up/Down= piston u ± {PISTON_U_STEP_M_S*100:.2f} cm/s"
        )
        help_lines = help_text.splitlines()
        help_line_h = self._get_font(11).get_linesize()
        help_top = 8 + help_line_h * len(help_lines)
        self._draw_text_cached(
            "controls_help",
            help_text,
            SIM_X0,
            help_top,
            Colors.BLACK,
            font_size=12,
            multiline=True,
            anchor_y="top",
        )

        if not self.running:
            self._draw_text_cached("paused_label", "PAUSED", SIM_X0 + 300, SIM_Y1 - 40, Colors.RED, 24)

        self._draw_text_cached(
            "credit_line",
            CREDIT_LINE,
            SCREEN_W - 12,
            SCREEN_H - 10,
            Colors.DARK_GRAY,
            font_size=11,
            anchor_x="right",
            anchor_y="top",
        )

        self.draw_particle_prompt()

    # -------------------------
    # Input
    # -------------------------
    def on_text(self, text: str):
        if self._prompt_active:
            self._prompt_error = ""
            if text.isdigit():
                self._prompt_text += text
                self._request_draw()
                return
            if text == "." and self._prompt_mode == "mols" and "." not in self._prompt_text:
                self._prompt_text += text
                self._request_draw()
                return
            return

        if self._focused_input_key is None:
            return

        allowed = set("0123456789.-+eE")
        for ch in text:
            if ch in allowed:
                self.target_inputs[self._focused_input_key].handle_text(ch)

    def on_key_press(self, key, modifiers):
        if self._prompt_active:
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._submit_particle_prompt(canceled=False)
            elif key == pygame.K_ESCAPE:
                self._submit_particle_prompt(canceled=True)
            elif key == pygame.K_BACKSPACE:
                self._prompt_text = self._prompt_text[:-1]
                self._prompt_error = ""
            elif key == pygame.K_m:
                self._prompt_mode = "mols"
            elif key == pygame.K_p:
                self._prompt_mode = "particles"
            self._request_draw()
            return

        if key == pygame.K_SPACE:
            self.running = not self.running
            self._request_draw()
            return

        if self._focused_input_key is not None:
            if key == pygame.K_BACKSPACE:
                self.target_inputs[self._focused_input_key].handle_backspace()
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE):
                self._set_input_focus(None)

        # any other key press pauses the simulation
        self.running = False
        self._request_draw()

        if key == pygame.K_r:
            self._begin_particle_prompt("reset", resume_running=False)
        elif key == pygame.K_a:
            self.state.thermal_on = not self.state.thermal_on
        elif key == pygame.K_LEFT:
            self.state.bath_Tk = max(1.0, self.state.bath_Tk - 25.0)
        elif key == pygame.K_RIGHT:
            self.state.bath_Tk = min(2000.0, self.state.bath_Tk + 25.0)
        elif key == pygame.K_UP:
            self.state.piston_u += PISTON_U_STEP_PX
        elif key == pygame.K_DOWN:
            self.state.piston_u -= PISTON_U_STEP_PX
        elif key == pygame.K_q:
            self.state.bath_Tk = 250.0
        elif key == pygame.K_w:
            self.state.bath_Tk = 273.16
        elif key == pygame.K_e:
            self.state.bath_Tk = 350.0

    def _present_frame(self):
        if self._display is None:
            return
        dst_size = self._display.get_size()
        if dst_size == (SCREEN_W, SCREEN_H):
            self._display.blit(self.screen, (0, 0))
        else:
            self._display.blit(pygame.transform.smoothscale(self.screen, dst_size), (0, 0))

    def _to_arcade_coords(self, pos):
        x, y = pos
        if self._display is not None:
            dw, dh = self._display.get_size()
            x = x * SCREEN_W / max(1, dw)
            y = y * SCREEN_H / max(1, dh)
        return x, SCREEN_H - y

    def _handle_mouse_down(self, pos, button):
        if button != 1 or self._prompt_active:
            return
        mx, my = self._to_arcade_coords(pos)
        if self.playback_slider.handle_mouse_down(mx, my):
            self._set_input_focus(None)
            self._request_draw()
            return
        for key, box in self.target_inputs.items():
            if box.contains(mx, my):
                self._set_input_focus(key)
                self._request_draw()
                return
        self._set_input_focus(None)

    def _handle_mouse_up(self, pos, button):
        if button != 1:
            return
        mx, my = self._to_arcade_coords(pos)
        self.playback_slider.handle_mouse_up(mx, my)
        self._request_draw()

    def _handle_mouse_motion(self, pos):
        if self._prompt_active:
            return
        mx, my = self._to_arcade_coords(pos)
        was_dragging = self.playback_slider.dragging
        self.playback_slider.handle_mouse_motion(mx, my)
        if was_dragging or self.playback_slider.dragging:
            self._request_draw()

    def run(self):
        pygame.key.start_text_input()
        running_main = True
        while running_main:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running_main = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_down(event.pos, event.button)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self._handle_mouse_up(event.pos, event.button)
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_motion(event.pos)
                elif event.type == pygame.TEXTINPUT:
                    self.on_text(event.text)
                elif event.type == pygame.KEYDOWN:
                    self.on_key_press(event.key, event.mod)

            self.on_update(dt)
            if self._should_draw():
                self.on_draw()
                self._present_frame()
                pygame.display.flip()

        if self._fast_mode_active:
            self._flush_pending_diag(self._effective_intervals()["hist_every"])

        if self.diag._log_writer_gas:
            self.diag._log_writer_gas.close()
        if self.diag._log_writer_particle:
            self.diag._log_writer_particle.close()
        pygame.quit()


# -------------------------
# Entrypoint
# -------------------------
def main():
    _set_windows_dpi_aware()
    app = CarnotApp()
    app.run()


if __name__ == "__main__":
    main()