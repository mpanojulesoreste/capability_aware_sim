"""
stigmergic.py — Decentralized swarm-style allocator and panel.

No central controller. Each StigmergicRobot runs purely local rules:
  1. Sense items and robots within SENSE_RADIUS.
  2. Filter: only consider items no closer robot (lower-ID tiebreak) is pursuing.
  3. Commit to nearest available item if idle.
  4. On delivery, steer toward user's reachable side if within BEACON_RADIUS.

Global information is not used. If the robot is outside BEACON_RADIUS it knows
the user's *position* (visible landmark) but not their reach profile.
"""

import numpy as np
import pygame
from collections import deque

from agents import RobotAgent, UserAgent, pos_m2p, m2p, REACH_RIGHT, REACH_LEFT, ROBOT_RADIUS_M
from tasks  import Task
from world  import PIXELS_PER_METER

SENSE_RADIUS  = 2.0
BEACON_RADIUS = 1.5
AVOID_DIST    = 0.50
HANDOFF_DIST  = 0.25

STATE_EXPLORING   = "exploring"
STATE_COMMITTED   = "committed"
STATE_CARRYING    = "carrying"
STATE_APPROACHING = "approaching"
STATE_HANDING_OFF = "handing_off"

SENSE_RING_ALPHA  = 60
BEACON_RING_ALPHA = 50
BEACON_COLOR      = (244, 213, 141)
COMMIT_LINE_LIFE  = 0.8
DEFER_X_LIFE      = 0.3
BEACON_FLASH_LIFE = 0.4


class StigmergicRobot(RobotAgent):
    """RobotAgent extended with local decision-making."""

    def __init__(self, robot_id: int, start_pos, rng=None):
        super().__init__(robot_id, start_pos)
        self.state        = STATE_EXPLORING
        self.committed_task    = None
        self.carrying_task     = None
        self.in_beacon         = False
        self.beacon_flash      = 0.0
        self._rng              = rng if rng is not None else np.random.default_rng()
        self.visible_task_ids  = []
        self.visible_robot_ids = []
        self._explore_target   = None
        self._waypoint_idx     = int(self._rng.integers(0, 5))

    def local_tick(self, dt, all_robots, all_tasks, user, sim_time,
                   commit_events, defer_events):
        dist_to_user = np.linalg.norm(self.pos - user.pos)
        was_in_beacon  = self.in_beacon
        self.in_beacon = dist_to_user <= BEACON_RADIUS
        if self.in_beacon and not was_in_beacon:
            self.beacon_flash = BEACON_FLASH_LIFE
        if self.beacon_flash > 0:
            self.beacon_flash = max(0.0, self.beacon_flash - dt)

        self.visible_task_ids  = [
            t.task_id for t in all_tasks
            if np.linalg.norm(t.pickup_pos - self.pos) <= SENSE_RADIUS
            and t.status not in ("done",)
        ]
        if self.committed_task is not None and \
                self.committed_task.task_id not in self.visible_task_ids:
            if self.committed_task.status not in ("done",):
                self.visible_task_ids.append(self.committed_task.task_id)

        self.visible_robot_ids = [
            r.id for r in all_robots
            if r is not self
            and np.linalg.norm(r.pos - self.pos) <= SENSE_RADIUS
        ]

        if self.state == STATE_EXPLORING:
            self._rule_explore(dt, all_robots, all_tasks, user,
                               commit_events, defer_events)

        elif self.state == STATE_COMMITTED:
            self._rule_committed(dt, all_robots, all_tasks, user,
                                 commit_events, defer_events)

        elif self.state in (STATE_CARRYING, STATE_APPROACHING):
            self._rule_carry(dt, user)

        elif self.state == STATE_HANDING_OFF:
            self._rule_handoff(dt, user)

        self._move_with_avoidance(dt, all_robots)

    def _rule_explore(self, dt, all_robots, all_tasks, user,
                      commit_events, defer_events):
        candidate = self._find_best_item(all_robots, all_tasks)
        if candidate is not None:
            self.committed_task = candidate
            candidate.status    = "assigned"
            candidate.assigned_robot = self.id
            self.state          = STATE_COMMITTED
            self.target         = candidate.pickup_pos.copy()
            commit_events.append((candidate.pickup_pos.copy(), self.color[:]))
        else:
            self._wander(dt)

    def _rule_committed(self, dt, all_robots, all_tasks, user,
                         commit_events, defer_events):
        """Heading to pickup. Recheck if a closer robot has appeared."""
        task = self.committed_task
        if task is None or task.status == "done":
            self._reset_to_explore()
            return

        closer = self._closer_robot_exists(task, all_robots)
        if closer:
            task.status = "pending"
            task.assigned_robot = None
            defer_events.append((task.pickup_pos.copy(), 0.3))
            self._reset_to_explore()
            return

        if self.at_target:
            self.carrying_task      = task
            self.committed_task     = None
            task.status             = "fetching"
            self.state              = STATE_CARRYING
            self._set_delivery_target(user)

    def _rule_carry(self, dt, user):
        """Carrying an item toward handoff. Adjust target as beacon info arrives."""
        if self.at_target:
            self.state = STATE_APPROACHING

        if self.in_beacon:
            self._set_delivery_target(user)
            self.state = STATE_APPROACHING

        dist_to_user = np.linalg.norm(self.pos - user.pos)
        if dist_to_user <= HANDOFF_DIST * 3:
            self.state = STATE_HANDING_OFF
            self.target = self.carrying_task.handoff_pos.copy()

    def _rule_handoff(self, dt, user):
        pass

    def _find_best_item(self, all_robots, all_tasks):
        """Return nearest visible pending item that no closer robot is pursuing."""
        visible_pending = [
            t for t in all_tasks
            if t.task_id in self.visible_task_ids and t.status == "pending"
        ]
        if not visible_pending:
            return None

        visible_pending.sort(key=lambda t: np.linalg.norm(t.pickup_pos - self.pos))

        for task in visible_pending:
            if not self._closer_robot_exists(task, all_robots):
                return task
        return None

    def _closer_robot_exists(self, task, all_robots):
        """
        True if another robot is observably closer to task.pickup_pos AND
        either committed to it or heading toward it.
        Tiebreaker: lower robot ID wins.
        """
        my_dist = np.linalg.norm(task.pickup_pos - self.pos)
        for other in all_robots:
            if other is self:
                continue
            if other.id not in self.visible_robot_ids:
                continue
            other_dist = np.linalg.norm(task.pickup_pos - other.pos)
            if other_dist < my_dist:
                return True
            if abs(other_dist - my_dist) < 0.05 and other.id < self.id:
                return True
        return False

    def _set_delivery_target(self, user):
        """Choose handoff point — capability-aware only if in beacon range."""
        task = self.carrying_task
        if task is None:
            return

        if self.in_beacon:
            h_rad = np.radians(user.heading_deg)
            right = np.array([np.cos(h_rad - np.pi/2),
                               np.sin(h_rad - np.pi/2)])
            task.handoff_pos = user.pos + right * REACH_RIGHT * 0.88
        else:
            task.handoff_pos = user.pos.copy()

        self.target = task.handoff_pos.copy()

    def _wander(self, dt):
        """Move toward a persistent explore waypoint; pick a new one on arrival."""
        waypoints = [
            np.array([4.0, 0.5]), np.array([4.0, 3.5]),
            np.array([2.5, 2.0]), np.array([1.5, 0.5]),
            np.array([1.5, 3.5]),
        ]
        if self._explore_target is None:
            self._explore_target = waypoints[self._waypoint_idx % len(waypoints)].copy()

        if np.linalg.norm(self.pos - self._explore_target) < 0.15:
            self._waypoint_idx = int(self._rng.integers(0, len(waypoints)))
            self._explore_target = waypoints[self._waypoint_idx].copy()

        self.target = self._explore_target

    def _move_with_avoidance(self, dt, all_robots):
        pass

    def _reset_to_explore(self):
        self.committed_task = None
        self.state          = STATE_EXPLORING
        self.target         = None

    @property
    def swarm_state(self):
        return self.state



class StigmergicPanelState:
    """
    Panel that runs StigmergicRobots with local rules instead of a central
    allocator. Structurally mirrors PanelState but overrides update() and draw().
    """

    def __init__(self, label: str, panel_x: int, scale: float = 1.0,
                 sidebar_w_override: int = None, seed: int = None):
        from metrics import MetricsLog
        self.label     = label
        self.panel_x   = panel_x
        self.scale     = scale
        self._pw       = int(500 * scale)
        self._ph       = int(400 * scale)
        self._sb       = sidebar_w_override if sidebar_w_override is not None \
                         else int(220 * scale)
        self.sidebar_x = panel_x + self._pw

        master_rng = np.random.default_rng(seed)
        robot_seeds = master_rng.integers(0, 2**31, size=3).tolist()

        starts = [(0.4, 0.4), (0.4, 3.6), (2.5, 2.0)]
        self.robots  = [
            StigmergicRobot(i, starts[i],
                            rng=np.random.default_rng(robot_seeds[i]))
            for i in range(3)
        ]
        self.user    = UserAgent((1.0, 2.0), heading_deg=0.0)
        from tasks import make_task_list
        self.tasks   = make_task_list()
        self.metrics = MetricsLog(label)

        self.sim_time  = 0.0
        self.started   = False
        self.done      = False
        self._font     = None

        self.flash_pos    = {}
        self.commit_lines = []
        self.defer_Xs     = []
        self.handoff_fx   = []

    def _px(self, val_m):
        return int(val_m * PIXELS_PER_METER * self.scale)

    def _pos_px(self, pos_m):
        return (self.panel_x + self._px(pos_m[0]),
                self._px(pos_m[1]))

    def _is_reachable(self, pos):
        delta = pos - self.user.pos
        dist  = np.linalg.norm(delta)
        h_rad = np.radians(self.user.heading_deg)
        right = np.array([np.cos(h_rad - np.pi/2),
                           np.sin(h_rad - np.pi/2)])
        reach = REACH_RIGHT if np.dot(delta, right) >= 0 else REACH_LEFT
        return dist <= reach

    def update(self, dt: float):
        if self.done:
            return
        self.sim_time += dt
        if not self.started:
            self.metrics.set_start(self.sim_time)
            self.started = True

        self.user.update(dt)

        commit_events = []
        defer_events  = []

        for robot in self.robots:
            robot.update(dt, self.robots, self.sim_time)
            robot.local_tick(dt, self.robots, self.tasks, self.user,
                             self.sim_time, commit_events, defer_events)

        for pos, color in commit_events:
            self.commit_lines.append([pos, color, COMMIT_LINE_LIFE])
        for pos, life in defer_events:
            self.defer_Xs.append([pos, life])

        self.commit_lines = [[p,c,t-dt] for p,c,t in self.commit_lines if t-dt > 0]
        self.defer_Xs     = [[p,t-dt]   for p,t in self.defer_Xs        if t-dt > 0]
        for tid in list(self.flash_pos):
            self.flash_pos[tid][1] -= dt
            if self.flash_pos[tid][1] <= 0:
                del self.flash_pos[tid]
        self.handoff_fx = [(p,st,oc) for p,st,oc in self.handoff_fx
                           if self.sim_time - st < 0.5]

        for robot in self.robots:
            if not robot.at_target:
                continue

            task = robot.carrying_task
            if task is not None and robot.state in (STATE_HANDING_OFF, STATE_APPROACHING):
                reachable = self._is_reachable(task.handoff_pos)
                if not reachable:
                    nudge = task.handoff_pos - np.array([0.25, 0.0])
                    self.user.reposition(nudge)
                    self.metrics.log_reposition()

                task.status    = "done"
                task.done_time = self.sim_time
                self.metrics.log_delivery(self.sim_time, reachable)
                self.flash_pos[task.task_id] = [task.handoff_pos.copy(), 0.7]
                outcome = "reachable" if reachable else "unreachable"
                self.handoff_fx.append((task.handoff_pos.copy(),
                                        self.sim_time, outcome))

                robot.carrying_task = None
                robot.state         = STATE_EXPLORING
                robot.target        = None

        if all(t.status == "done" for t in self.tasks):
            self.done = True

    def draw(self, surface, font):
        self._font = font
        self._draw_bg(surface)
        self._draw_sensing_circles(surface)
        self._draw_beacon_zone(surface)
        self._draw_conflict_zones(surface)
        self._draw_items(surface)
        self._draw_handoff_markers(surface)
        self._draw_flash(surface)
        self._draw_commit_lines(surface)
        self._draw_defer_Xs(surface)
        self._draw_handoff_fx(surface)
        self.user.draw(surface, self.panel_x)
        for robot in self.robots:
            robot.draw(surface, self.panel_x, self.sim_time)
            self._draw_beacon_indicator(surface, robot)
        self._draw_sidebar(surface)

    def _draw_bg(self, surface):
        from world import ROOM_W, ROOM_H
        PANEL_BG  = ( 40,  42,  52)
        GRID_COLOR= ( 52,  55,  66)
        TEXT_COLOR= (210, 215, 225)
        pygame.draw.rect(surface, PANEL_BG,
                         pygame.Rect(self.panel_x, 0, self._pw, self._ph))
        for gx in range(int(ROOM_W) + 1):
            px = self.panel_x + self._px(gx)
            pygame.draw.line(surface, GRID_COLOR, (px, 0), (px, self._ph))
        for gy in range(int(ROOM_H) + 1):
            py = self._px(gy)
            pygame.draw.line(surface, GRID_COLOR,
                             (self.panel_x, py), (self.panel_x + self._pw, py))
        surf = self._font.render(self.label, True, TEXT_COLOR)
        surface.blit(surf, (self.panel_x + 4, 4))
        if self.done:
            s = self._font.render(
                f"DONE {self.metrics.total_time():.1f}s", True, (80, 210, 100))
            surface.blit(s, (self.panel_x + self._pw - 100, 4))

    def _draw_sensing_circles(self, surface):
        """V1: dashed sensing radius ring around each robot."""
        for robot in self.robots:
            cx, cy = self._pos_px(robot.pos)
            radius = int(self._px(SENSE_RADIUS))
            rc, gc, bc = robot.color
            self._draw_dashed_circle(surface, (cx, cy), radius,
                                     (rc, gc, bc, SENSE_RING_ALPHA), dash_len=8)

    def _draw_beacon_zone(self, surface):
        """V2: dashed beacon range circle around user."""
        cx, cy = self._pos_px(self.user.pos)
        r = int(self._px(BEACON_RADIUS))
        self._draw_dashed_circle(surface, (cx, cy), r,
                                 (*BEACON_COLOR, BEACON_RING_ALPHA), dash_len=10)
        lbl = self._font.render("beacon", True, BEACON_COLOR)
        surface.blit(lbl, (cx - lbl.get_width()//2, cy - r - 14))

    def _draw_beacon_indicator(self, surface, robot):
        """V3: persistent yellow dot + flash ring when entering beacon."""
        cx, cy = self._pos_px(robot.pos)
        if robot.in_beacon:
            pygame.draw.circle(surface, BEACON_COLOR, (cx - 4, cy - 4), 3)
        if robot.beacon_flash > 0:
            alpha = int(255 * robot.beacon_flash / BEACON_FLASH_LIFE)
            tmp = pygame.Surface((60, 60), pygame.SRCALPHA)
            pygame.draw.circle(tmp, (*BEACON_COLOR, alpha), (30, 30), 28, 3)
            surface.blit(tmp, (cx - 30, cy - 30))

    def _draw_conflict_zones(self, surface):
        """V6: faint red shading at the midpoint when two robots share a visible item."""
        robots = self.robots
        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                ra, rb = robots[i], robots[j]
                dist = np.linalg.norm(ra.pos - rb.pos)
                if dist > SENSE_RADIUS * 2:
                    continue
                shared = set(ra.visible_task_ids) & set(rb.visible_task_ids)
                if not shared:
                    continue
                mid = (ra.pos + rb.pos) / 2
                cx, cy = self._pos_px(mid)
                blob_r = 18
                tmp = pygame.Surface((blob_r*2+4, blob_r*2+4), pygame.SRCALPHA)
                pygame.draw.circle(tmp, (220, 60, 60, 40),
                                   (blob_r+2, blob_r+2), blob_r)
                surface.blit(tmp, (cx - blob_r - 2, cy - blob_r - 2))

    def _draw_commit_lines(self, surface):
        """V4: ephemeral line from robot to item on commitment."""
        for entry in self.commit_lines:
            pos, color, remaining = entry
            alpha = int(180 * remaining / COMMIT_LINE_LIFE)
            robot = min(self.robots,
                        key=lambda r: np.linalg.norm(r.pos - pos))
            rx, ry = self._pos_px(robot.pos)
            ix, iy = self._pos_px(pos)
            tmp = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            pygame.draw.line(tmp, (*color, alpha), (rx, ry), (ix, iy), 2)
            surface.blit(tmp, (0, 0))

    def _draw_defer_Xs(self, surface):
        """V5: gray X flash at item when a robot defers."""
        for entry in self.defer_Xs:
            pos, remaining = entry
            alpha = int(200 * remaining / DEFER_X_LIFE)
            ix, iy = self._pos_px(pos)
            tmp = pygame.Surface((20, 20), pygame.SRCALPHA)
            c = (160, 160, 160, alpha)
            pygame.draw.line(tmp, c, (2, 2), (18, 18), 2)
            pygame.draw.line(tmp, c, (18, 2), (2, 18), 2)
            surface.blit(tmp, (ix - 10, iy - 10))

    def _draw_handoff_fx(self, surface):
        """Expanding ring at handoff point (green=reachable, red=unreachable)."""
        for pos, start_t, outcome in self.handoff_fx:
            elapsed = self.sim_time - start_t
            if elapsed >= 0.5:
                continue
            frac   = elapsed / 0.5
            radius = int(self._px(0.05 + frac * 0.20))
            alpha  = int(255 * (1 - frac))
            width  = max(1, int(3 * (1 - frac)))
            color  = (106, 176, 76) if outcome == "reachable" else (231, 76, 60)
            tmp = pygame.Surface((radius*2+4, radius*2+4), pygame.SRCALPHA)
            pygame.draw.circle(tmp, (*color, alpha), (radius+2, radius+2), radius, width)
            px, py = self._pos_px(pos)
            surface.blit(tmp, (px - radius - 2, py - radius - 2))

    def _draw_items(self, surface):
        DIM = (55, 58, 70)
        for task in self.tasks:
            px, py = self._pos_px(task.pickup_pos)
            side = max(5, int(8 * self.scale))
            color = task.color if task.status != "done" else DIM
            pygame.draw.rect(surface, color,
                             (px - side//2, py - side//2, side, side))
            if task.status != "done":
                pygame.draw.rect(surface, (200, 200, 200),
                                 (px - side//2, py - side//2, side, side), 1)
            lbl = self._font.render(str(task.task_id), True,
                                    (220, 220, 220) if task.status != "done" else DIM)
            surface.blit(lbl, (px + side//2 + 2, py - 6))

    def _draw_handoff_markers(self, surface):
        HANDOFF_COLOR = (255, 165, 70)
        for task in self.tasks:
            if task.status not in ("assigned", "fetching", "delivering"):
                continue
            px, py = self._pos_px(task.handoff_pos)
            pygame.draw.circle(surface, HANDOFF_COLOR, (px, py), 6, 2)
            pygame.draw.line(surface, HANDOFF_COLOR, (px-8, py), (px+8, py), 1)
            pygame.draw.line(surface, HANDOFF_COLOR, (px, py-8), (px, py+8), 1)

    def _draw_flash(self, surface):
        for tid, (pos, remaining) in self.flash_pos.items():
            alpha = remaining / 0.7
            px, py = self._pos_px(pos)
            r = int(10 + (1 - alpha) * 18)
            c = (int(255 * alpha), int(240 * alpha), int(80 * alpha))
            pygame.draw.circle(surface, c, (px, py), r, 2)

    def _draw_sidebar(self, surface):
        SIDEBAR_BG = ( 26,  28,  36)
        GRID_COLOR = ( 52,  55,  66)
        TEXT_COLOR = (210, 215, 225)
        DIM_COLOR  = (120, 125, 140)

        sb_rect = pygame.Rect(self.sidebar_x, 0, self._sb, self._ph)
        pygame.draw.rect(surface, SIDEBAR_BG, sb_rect)
        pygame.draw.line(surface, GRID_COLOR,
                         (self.sidebar_x, 0), (self.sidebar_x, self._ph), 1)

        m  = self.metrics
        FS = max(10, int(self._font.size("A")[1]))
        y  = 6

        def row(text, color):
            nonlocal y
            surf = self._font.render(text, True, color)
            surface.blit(surf, (self.sidebar_x + 4, y))
            y += FS + 2

        row("=== METRICS ===", DIM_COLOR)
        row(f"Time: {self.sim_time:.1f}s",         TEXT_COLOR)
        row(f"Deliveries:  {m.delivery_count}",    (80, 210, 100))
        row(f"Repositions: {m.reposition_count}",  (230, 160, 60))
        row(f"Unreachable: {m.unreachable_count}", (220, 80, 80))
        y += 4

        row("=== ROBOTS ===", DIM_COLOR)
        for robot in self.robots:
            in_b = "B" if robot.in_beacon else " "
            vis_t = ",".join(str(i) for i in robot.visible_task_ids) or "-"
            vis_r = ",".join(str(i) for i in robot.visible_robot_ids) or "-"
            row(f" R{robot.id}[{in_b}] {robot.swarm_state[:9]}", robot.color)
            row(f"   items:{vis_t}", DIM_COLOR)
            row(f"   bots:{vis_r}",  DIM_COLOR)
        y += 4

        row("=== TASKS ===", DIM_COLOR)
        for task in self.tasks:
            row(f" T{task.task_id}: {task.status}", task.color)

    def _draw_dashed_circle(self, surface, center, radius, rgba,
                             dash_len=8, gap_len=6):
        """Dashed circle drawn on a local SRCALPHA surface sized to the circle."""
        if radius <= 2:
            return
        pad = 2
        size = (radius + pad) * 2
        tmp  = pygame.Surface((size, size), pygame.SRCALPHA)
        lc = (radius + pad, radius + pad)
        rect = pygame.Rect(pad, pad, radius * 2, radius * 2)

        circ    = 2 * np.pi * radius
        n_dash  = max(1, int(circ / (dash_len + gap_len)))
        dash_a  = 2 * np.pi * (dash_len / circ)

        for i in range(n_dash):
            start_a = 2 * np.pi * i / n_dash
            end_a   = start_a + dash_a
            pygame.draw.arc(tmp, rgba, rect, start_a,
                            min(end_a, start_a + dash_a), 1)

        surface.blit(tmp, (center[0] - radius - pad, center[1] - radius - pad))
