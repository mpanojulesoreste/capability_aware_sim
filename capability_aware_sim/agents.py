"""Robot and user agents: movement, reach geometry, and rendering."""

import numpy as np
import pygame
from collections import deque
from world import PIXELS_PER_METER

REACH_RIGHT   = 0.8
REACH_LEFT    = 0.25
MOVE_SPEED    = 0.4
FOV_ANGLE_DEG = 120.0

ROBOT_MAX_SPEED  = 0.6
ROBOT_RADIUS_M   = 0.12
AVOIDANCE_DIST   = 0.30

ROBOT_COLORS = [
    (220,  80,  80),
    ( 80, 160, 220),
    ( 80, 210, 130),
]
USER_COLOR      = (230, 200,  50)
REACH_R_COLOR   = ( 80, 200,  80, 55)
REACH_L_COLOR   = (200, 130,  50, 45)
FOV_COLOR       = (200, 200, 255, 30)


def m2p(val):
    """Meters to pixels (scalar or array)."""
    return val * PIXELS_PER_METER


def pos_m2p(pos_m, panel_offset_x=0):
    """Convert (x,y) meters → (px, py) pixels with optional panel x offset."""
    x = int(pos_m[0] * PIXELS_PER_METER) + panel_offset_x
    y = int(pos_m[1] * PIXELS_PER_METER)
    return (x, y)


TRAIL_DURATION = 2.0
TRAIL_MAXLEN   = 120


class RobotAgent:
    def __init__(self, robot_id: int, start_pos):
        self.id       = robot_id
        self.pos      = np.array(start_pos, dtype=float)
        self.vel      = np.zeros(2)
        self.color    = ROBOT_COLORS[robot_id % len(ROBOT_COLORS)]
        self.state    = "idle"
        self.task     = None
        self.target   = None
        self.trail    = deque(maxlen=TRAIL_MAXLEN)

    def set_target(self, pos):
        self.target = np.array(pos, dtype=float)

    def update(self, dt, others, sim_time=0.0):
        self.trail.append((self.pos.copy(), sim_time))
        while self.trail and sim_time - self.trail[0][1] > TRAIL_DURATION:
            self.trail.popleft()

        if self.target is None:
            self.vel[:] = 0
            return

        to_target = self.target - self.pos
        dist = np.linalg.norm(to_target)

        if dist < 0.02:
            self.pos = self.target.copy()
            self.vel[:] = 0
            self.target = None
            return

        desired = (to_target / dist) * ROBOT_MAX_SPEED

        sep = np.zeros(2)
        for other in others:
            if other is self:
                continue
            delta = self.pos - other.pos
            d = np.linalg.norm(delta)
            if 0 < d < AVOIDANCE_DIST:
                sep += (delta / d) * (AVOIDANCE_DIST - d)

        vel = desired + sep * 1.5
        speed = np.linalg.norm(vel)
        if speed > ROBOT_MAX_SPEED:
            vel = vel / speed * ROBOT_MAX_SPEED

        self.vel = vel
        self.pos += vel * dt

    @property
    def at_target(self):
        return self.target is None

    def draw(self, surface, panel_offset_x=0, sim_time=0.0):
        self._draw_trail(surface, panel_offset_x, sim_time)

        px, py = pos_m2p(self.pos, panel_offset_x)
        r = max(4, int(m2p(ROBOT_RADIUS_M)))
        pygame.draw.circle(surface, self.color, (px, py), r)
        pygame.draw.circle(surface, (255, 255, 255), (px, py), r, 1)

        font = pygame.font.SysFont("monospace", 10)
        label = font.render(str(self.id), True, (255, 255, 255))
        surface.blit(label, (px - 4, py - 5))

    def _draw_trail(self, surface, panel_offset_x, sim_time):
        pts = list(self.trail)
        if len(pts) < 2:
            return
        r, g, b = self.color
        for i in range(1, len(pts)):
            pos_a, _   = pts[i - 1]
            pos_b, t_b = pts[i]
            age   = sim_time - t_b
            alpha = int(150 * max(0.0, 1.0 - age / TRAIL_DURATION))
            if alpha <= 0:
                continue
            color = (r, g, b, alpha)
            xa, ya = pos_m2p(pos_a, panel_offset_x)
            xb, yb = pos_m2p(pos_b, panel_offset_x)
            x0, y0 = min(xa, xb) - 2, min(ya, yb) - 2
            x1, y1 = max(xa, xb) + 2, max(ya, yb) + 2
            w, h   = max(x1 - x0, 1), max(y1 - y0, 1)
            tmp    = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.line(tmp, color,
                             (xa - x0, ya - y0), (xb - x0, yb - y0), 2)
            surface.blit(tmp, (x0, y0))


class UserAgent:
    def __init__(self, home_pos, heading_deg=0.0):
        self.pos         = np.array(home_pos, dtype=float)
        self.home        = np.array(home_pos, dtype=float)
        self.heading_deg = heading_deg
        self.prev_pos    = None
        self.arrow_timer = 0.0

    def reposition(self, new_pos):
        self.prev_pos    = self.pos.copy()
        self.pos         = np.array(new_pos, dtype=float)
        self.arrow_timer = 1.2

    def update(self, dt):
        if self.arrow_timer > 0:
            self.arrow_timer -= dt

    def draw(self, surface, panel_offset_x=0):
        px, py = pos_m2p(self.pos, panel_offset_x)

        self._draw_fov(surface, px, py)
        self._draw_reach(surface, px, py)

        pygame.draw.circle(surface, USER_COLOR, (px, py), 10)
        pygame.draw.circle(surface, (255, 255, 255), (px, py), 10, 2)

        if self.prev_pos is not None and self.arrow_timer > 0:
            ox, oy = pos_m2p(self.prev_pos, panel_offset_x)
            alpha  = min(1.0, self.arrow_timer / 0.4)
            color  = (255, int(60 * alpha), int(60 * alpha))
            pygame.draw.line(surface, color, (ox, oy), (px, py), 2)
            pygame.draw.circle(surface, color, (px, py), 4)

    def _draw_reach(self, surface, px, py):
        h = self.heading_deg
        self._draw_filled_arc(surface, px, py, REACH_RIGHT,
                               h - 90, h + 90, (60, 200, 90, 60))
        self._draw_filled_arc(surface, px, py, REACH_LEFT,
                               h + 90, h + 270, (200, 140, 50, 50))
        self._draw_arc_outline(surface, px, py, REACH_RIGHT,
                                h - 90, h + 90, (80, 220, 110))
        self._draw_arc_outline(surface, px, py, REACH_LEFT,
                                h + 90, h + 270, (210, 160, 70))

    def _draw_filled_arc(self, surface, cx, cy, radius_m,
                          start_deg, end_deg, rgba):
        """Filled pie slice drawn on a per-call SRCALPHA surface then blitted."""
        r   = int(m2p(radius_m))
        pad = 2
        sz  = (r + pad) * 2
        tmp = pygame.Surface((sz, sz), pygame.SRCALPHA)
        tmp.fill((0, 0, 0, 0))

        # Build polygon: center + arc points
        cx_l, cy_l = r + pad, r + pad
        pts = [(cx_l, cy_l)]
        steps = 24
        a0 = np.radians(start_deg)
        a1 = np.radians(end_deg)
        for i in range(steps + 1):
            a   = a0 + (a1 - a0) * i / steps
            pts.append((cx_l + r * np.cos(a), cy_l + r * np.sin(a)))

        if len(pts) >= 3:
            pygame.draw.polygon(tmp, rgba, pts)
        surface.blit(tmp, (cx - r - pad, cy - r - pad))

    def _draw_arc_outline(self, surface, cx, cy, radius_m,
                           start_deg, end_deg, rgb):
        r    = int(m2p(radius_m))
        rect = pygame.Rect(cx - r, cy - r, r * 2, r * 2)
        a_start = np.radians(-end_deg)
        a_end   = np.radians(-start_deg)
        pygame.draw.arc(surface, rgb, rect, a_start, a_end, 1)

    def _draw_fov(self, surface, px, py):
        half = FOV_ANGLE_DEG / 2
        h_rad   = np.radians(self.heading_deg)
        l_rad   = np.radians(self.heading_deg - half)
        r_rad   = np.radians(self.heading_deg + half)
        length  = int(m2p(1.5))

        tip_l = (px + int(np.cos(l_rad) * length),
                 py + int(np.sin(l_rad) * length))
        tip_r = (px + int(np.cos(r_rad) * length),
                 py + int(np.sin(r_rad) * length))

        tmp = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        tmp.fill((0, 0, 0, 0))
        pygame.draw.polygon(tmp, (200, 200, 255, 28),
                            [(px, py), tip_l, tip_r])
        surface.blit(tmp, (0, 0))
