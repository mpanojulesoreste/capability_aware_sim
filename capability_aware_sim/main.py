"""Capability-Aware Multi-Robot Simulation — pygame entry point."""

import sys
import pygame
import numpy as np
import matplotlib
matplotlib.use("Agg")

from world import (
    ROOM_W, ROOM_H, PANEL_W, PANEL_H, SIDEBAR_W,
    USER_HOME, HANDOFF_BASE,
)
from agents import RobotAgent, UserAgent, pos_m2p, m2p, REACH_RIGHT, REACH_LEFT
from tasks  import make_task_list
from allocators import baseline_allocator, adaptive_allocator
from metrics import MetricsLog, plot_summary
from stigmergic import StigmergicPanelState

FPS       = 60
SIM_SPEED = 1.0
FONT_SIZE = 13

GAP      = 12
WINDOW_W = (PANEL_W + SIDEBAR_W) * 2 + GAP
WINDOW_H = PANEL_H

BG_COLOR      = ( 28,  28,  34)
PANEL_BG      = ( 40,  42,  52)
SIDEBAR_BG    = ( 26,  28,  36)
GRID_COLOR    = ( 52,  55,  66)
TEXT_COLOR    = (210, 215, 225)
DIM_COLOR     = (120, 125, 140)
HANDOFF_COLOR = (255, 165,  70)
FLASH_COLOR   = (255, 240, 100)



class PanelState:
    """All mutable state for one simulation panel."""

    def __init__(self, label: str, allocator_fn, panel_x: int):
        self.label        = label
        self.allocator_fn = allocator_fn
        self.panel_x      = panel_x
        self.sidebar_x    = panel_x + PANEL_W

        starts = [(0.4, 0.4), (0.4, 3.6), (2.5, 2.0)]
        self.robots  = [RobotAgent(i, starts[i]) for i in range(3)]
        self.user    = UserAgent(USER_HOME, heading_deg=0.0)
        self.tasks   = make_task_list()
        self.metrics = MetricsLog(label)

        self.sim_time    = 0.0
        self.started     = False
        self.done        = False
        self.flash_pos   = {}
        self._font       = None

    def _is_reachable(self, pos):
        delta = pos - self.user.pos
        dist  = np.linalg.norm(delta)
        h_rad = np.radians(self.user.heading_deg)
        right = np.array([ np.cos(h_rad - np.pi / 2),
                            np.sin(h_rad - np.pi / 2)])
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

        for robot in self.robots:
            robot.update(dt, self.robots, self.sim_time)

        for robot in self.robots:
            if not robot.at_target:
                continue

            if robot.state == "fetching":
                robot.state = "delivering"
                robot.set_target(robot.task.handoff_pos)
                robot.task.status = "delivering"

            elif robot.state == "delivering":
                task      = robot.task
                reachable = self._is_reachable(task.handoff_pos)

                if not reachable:
                    nudge = task.handoff_pos - np.array([0.25, 0.0])
                    self.user.reposition(nudge)
                    self.metrics.log_reposition()

                task.status    = "done"
                task.done_time = self.sim_time
                self.metrics.log_delivery(self.sim_time, reachable)
                self.flash_pos[task.task_id] = [task.handoff_pos.copy(), 0.7]

                robot.state = "idle"
                robot.task  = None

        for tid in list(self.flash_pos):
            self.flash_pos[tid][1] -= dt
            if self.flash_pos[tid][1] <= 0:
                del self.flash_pos[tid]

        result = self.allocator_fn(self.robots, self.tasks, self.user)
        if result is not None:
            robot, task, handoff   = result
            task.handoff_pos       = handoff
            task.status            = "assigned"
            task.start_time        = self.sim_time
            task.assigned_robot    = robot.id
            robot.task             = task
            robot.state            = "fetching"
            robot.set_target(task.pickup_pos)

        if all(t.status == "done" for t in self.tasks):
            self.done = True

    def draw(self, surface, font):
        self._font = font
        self._draw_bg(surface)
        self._draw_robot_paths(surface)
        self._draw_items(surface)
        self._draw_handoff_markers(surface)
        self._draw_flash(surface)
        self.user.draw(surface, self.panel_x)
        for robot in self.robots:
            robot.draw(surface, self.panel_x, self.sim_time)
        self._draw_sidebar(surface)

    def _draw_bg(self, surface):
        pygame.draw.rect(surface, PANEL_BG,
                         pygame.Rect(self.panel_x, 0, PANEL_W, PANEL_H))
        for gx in range(int(ROOM_W) + 1):
            px = self.panel_x + int(gx * m2p(1))
            pygame.draw.line(surface, GRID_COLOR, (px, 0), (px, PANEL_H))
        for gy in range(int(ROOM_H) + 1):
            py = int(gy * m2p(1))
            pygame.draw.line(surface, GRID_COLOR,
                             (self.panel_x, py), (self.panel_x + PANEL_W, py))

        label_surf = self._font.render(self.label, True, TEXT_COLOR)
        surface.blit(label_surf, (self.panel_x + 6, 5))

        if self.done:
            s = self._font.render(
                f"DONE  {self.metrics.total_time():.1f}s", True, (80, 210, 100))
            surface.blit(s, (self.panel_x + PANEL_W - 110, 5))

    def _draw_robot_paths(self, surface):
        for robot in self.robots:
            if robot.target is None:
                continue
            px0, py0 = pos_m2p(robot.pos,    self.panel_x)
            px1, py1 = pos_m2p(robot.target, self.panel_x)
            pygame.draw.line(surface, (70, 72, 88), (px0, py0), (px1, py1), 1)

    def _draw_items(self, surface):
        for task in self.tasks:
            px, py = pos_m2p(task.pickup_pos, self.panel_x)
            side   = 8
            color  = task.color if task.status != "done" else (55, 58, 70)
            pygame.draw.rect(surface, color,
                             (px - side//2, py - side//2, side, side))
            if task.status != "done":
                pygame.draw.rect(surface, (200, 200, 200),
                                 (px - side//2, py - side//2, side, side), 1)

            lbl = self._font.render(str(task.task_id), True,
                                    (220, 220, 220) if task.status != "done" else DIM_COLOR)
            surface.blit(lbl, (px + 5, py - 6))

    def _draw_handoff_markers(self, surface):
        for task in self.tasks:
            if task.status not in ("assigned", "fetching", "delivering"):
                continue
            px, py = pos_m2p(task.handoff_pos, self.panel_x)
            pygame.draw.circle(surface, HANDOFF_COLOR, (px, py), 7, 2)
            pygame.draw.line(surface, HANDOFF_COLOR, (px-9, py), (px+9, py), 1)
            pygame.draw.line(surface, HANDOFF_COLOR, (px, py-9), (px, py+9), 1)
            if task.status == "delivering":
                for robot in self.robots:
                    if robot.task is task:
                        rx, ry = pos_m2p(robot.pos, self.panel_x)
                        pygame.draw.line(surface, (160, 110, 40),
                                         (rx, ry), (px, py), 1)

    def _draw_flash(self, surface):
        for tid, (pos, remaining) in self.flash_pos.items():
            alpha = remaining / 0.7
            px, py = pos_m2p(pos, self.panel_x)
            r = int(10 + (1 - alpha) * 18)
            c = (int(255 * alpha), int(240 * alpha), int(80 * alpha))
            pygame.draw.circle(surface, c, (px, py), r, 2)

    def _draw_sidebar(self, surface):
        pygame.draw.rect(surface, SIDEBAR_BG,
                         pygame.Rect(self.sidebar_x, 0, SIDEBAR_W, PANEL_H))
        pygame.draw.line(surface, GRID_COLOR,
                         (self.sidebar_x, 0), (self.sidebar_x, PANEL_H), 1)

        m = self.metrics
        sections = [
            [("=== METRICS ===", DIM_COLOR),
             (f"Sim time:  {self.sim_time:5.1f}s",  TEXT_COLOR),
             (f"Deliveries:  {m.delivery_count}",   (80, 210, 100)),
             (f"Repositions: {m.reposition_count}", (230, 160, 60)),
             (f"Unreachable: {m.unreachable_count}",(220,  80, 80))],
            [("=== ROBOTS ===", DIM_COLOR)]
            + [(f"  R{r.id} {r.state}", r.color) for r in self.robots],
            [("=== TASKS ===", DIM_COLOR)]
            + [(f"  T{t.task_id}: {t.status}", t.color) for t in self.tasks],
        ]

        y = 8
        for section in sections:
            for text, color in section:
                surf = self._font.render(text, True, color)
                surface.blit(surf, (self.sidebar_x + 5, y))
                y += FONT_SIZE + 3
            y += 4



def show_summary_chart(panels: list):
    b, a = panels[0].metrics.summary(), panels[1].metrics.summary()
    plot_summary(b, a)


def show_summary_chart_three(panels: list):
    """Print all three metrics summaries to stdout for the stigmergic run."""
    for p in panels:
        s = p.metrics.summary()
        print(f"\n{s['label']}")
        print(f"  Total time:       {s['total_time']:.2f}s")
        print(f"  Repositions:      {s['repositions']}")
        print(f"  Unreachable:      {s['unreachable']}")
        print(f"  Deliveries:       {s['deliveries']}")


def run_standard():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(
        "Capability-Aware Multi-Robot Sim  |  Q quit  S screenshot  F fast")
    font  = pygame.font.SysFont("monospace", FONT_SIZE)
    clock = pygame.time.Clock()

    left_x  = 0
    right_x = PANEL_W + SIDEBAR_W + GAP
    panels  = [
        PanelState("Baseline  (capability-blind)",  baseline_allocator, left_x),
        PanelState("Adaptive  (capability-aware)",  adaptive_allocator,  right_x),
    ]

    sim_speed   = 1.0
    chart_shown = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_s:
                    pygame.image.save(screen, "screenshot.png")
                    print("Screenshot saved → screenshot.png")
                if event.key == pygame.K_f:
                    sim_speed = 3.0 if sim_speed == 1.0 else 1.0
                    print(f"Sim speed: {sim_speed}×")

        raw_dt = clock.tick(FPS) / 1000.0
        dt     = min(raw_dt, 0.05) * sim_speed

        for panel in panels:
            panel.update(dt)

        screen.fill(BG_COLOR)
        for panel in panels:
            panel.draw(screen, font)

        mid = PANEL_W + SIDEBAR_W + GAP // 2
        pygame.draw.line(screen, (60, 62, 74), (mid, 0), (mid, PANEL_H), 2)

        spd = font.render(f"{sim_speed:.0f}×", True,
                          (255, 200, 60) if sim_speed > 1 else DIM_COLOR)
        screen.blit(spd, (mid - 10, PANEL_H - 20))

        pygame.display.flip()

        if not chart_shown and all(p.done for p in panels):
            chart_shown = True
            show_summary_chart(panels)


def run_stigmergic():
    """Three panels: Baseline | Adaptive | Stigmergic, each rendered offscreen then scaled."""
    SLOT_W = 510
    WIN_W  = SLOT_W * 3 + 2 * 8
    WIN_H  = PANEL_H
    GAP3   = 8

    FULL_W_STD  = PANEL_W + SIDEBAR_W
    FULL_W_STIG = PANEL_W + 180

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(
        "3-Allocator Comparison  |  Q quit  S screenshot  F fast")
    font_big  = pygame.font.SysFont("monospace", FONT_SIZE)
    font_stig = pygame.font.SysFont("monospace", FONT_SIZE)
    clock     = pygame.time.Clock()

    surf_base  = pygame.Surface((FULL_W_STD,  WIN_H))
    surf_adapt = pygame.Surface((FULL_W_STD,  WIN_H))
    surf_stig  = pygame.Surface((FULL_W_STIG, WIN_H))

    panels_std = [
        PanelState("Baseline (blind)", baseline_allocator, 0),
        PanelState("Adaptive (aware)", adaptive_allocator, 0),
    ]
    panel_stig = StigmergicPanelState("Stigmergic (swarm)", 0,
                                       sidebar_w_override=180)

    sim_speed   = 1.0
    chart_shown = False

    # Screen x positions for each scaled slot
    x0 = 0
    x1 = SLOT_W + GAP3
    x2 = 2 * (SLOT_W + GAP3)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_s:
                    pygame.image.save(screen, "screenshot_stigmergic.png")
                    print("Screenshot saved → screenshot_stigmergic.png")
                if event.key == pygame.K_f:
                    sim_speed = 3.0 if sim_speed == 1.0 else 1.0

        raw_dt = clock.tick(FPS) / 1000.0
        dt     = min(raw_dt, 0.05) * sim_speed

        panels_std[0].update(dt)
        panels_std[1].update(dt)
        panel_stig.update(dt)

        screen.fill(BG_COLOR)

        surf_base.fill(BG_COLOR)
        panels_std[0].draw(surf_base, font_big)
        screen.blit(pygame.transform.scale(surf_base, (SLOT_W, WIN_H)), (x0, 0))

        surf_adapt.fill(BG_COLOR)
        panels_std[1].draw(surf_adapt, font_big)
        screen.blit(pygame.transform.scale(surf_adapt, (SLOT_W, WIN_H)), (x1, 0))

        surf_stig.fill(BG_COLOR)
        panel_stig.draw(surf_stig, font_stig)
        screen.blit(pygame.transform.scale(surf_stig, (SLOT_W, WIN_H)), (x2, 0))

        for xd in [x1 - GAP3//2, x2 - GAP3//2]:
            pygame.draw.line(screen, (60, 62, 74), (xd, 0), (xd, WIN_H), 2)

        spd = font_big.render(f"{sim_speed:.0f}x", True,
                              (255, 200, 60) if sim_speed > 1 else (80, 82, 96))
        screen.blit(spd, (WIN_W // 2 - 10, WIN_H - 20))

        pygame.display.flip()

        if not chart_shown and all(p.done for p in panels_std) and panel_stig.done:
            chart_shown = True
            show_summary_chart_three(
                [panels_std[0], panels_std[1], panel_stig])


def run_single(mode: str):
    from stigmergic import StigmergicPanelState

    WIN_W = PANEL_W + SIDEBAR_W
    WIN_H = PANEL_H

    pygame.init()
    pygame.display.set_caption(
        f"{mode.capitalize()}  |  Q quit  S screenshot  F fast")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    font   = pygame.font.SysFont("monospace", FONT_SIZE)
    clock  = pygame.time.Clock()

    if mode == "baseline":
        panel = PanelState("Baseline  (capability-blind)", baseline_allocator, 0)
    elif mode == "adaptive":
        panel = PanelState("Adaptive  (capability-aware)", adaptive_allocator, 0)
    elif mode == "stigmergic":
        panel = StigmergicPanelState("Stigmergic  (swarm)", 0,
                                      sidebar_w_override=SIDEBAR_W)
    else:
        print(f"Unknown mode '{mode}'. Choose: baseline | adaptive | stigmergic")
        sys.exit(1)

    sim_speed   = 1.0
    chart_shown = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_s:
                    fname = f"screenshot_{mode}.png"
                    pygame.image.save(screen, fname)
                    print(f"Screenshot saved → {fname}")
                if event.key == pygame.K_f:
                    sim_speed = 3.0 if sim_speed == 1.0 else 1.0
                    print(f"Sim speed: {sim_speed}×")

        raw_dt = clock.tick(FPS) / 1000.0
        dt     = min(raw_dt, 0.05) * sim_speed

        panel.update(dt)

        screen.fill(BG_COLOR)
        panel.draw(screen, font)

        spd = font.render(f"{sim_speed:.0f}x", True,
                          (255, 200, 60) if sim_speed > 1 else DIM_COLOR)
        screen.blit(spd, (WIN_W - 30, WIN_H - 20))

        pygame.display.flip()

        if not chart_shown and panel.done:
            chart_shown = True
            s = panel.metrics.summary()
            print(f"\n=== {s['label']} ===")
            print(f"  Total time:   {s['total_time']:.2f}s")
            print(f"  Deliveries:   {s['deliveries']}")
            print(f"  Repositions:  {s['repositions']}")
            print(f"  Unreachable:  {s['unreachable']}")
            if mode in ("baseline", "adaptive"):
                show_summary_chart([panel, panel])


def _get_flag(prefix: str) -> str | None:
    """Return the value of a --key=value CLI flag, or None."""
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def main():
    mode = _get_flag("--mode=")
    if mode:
        run_single(mode)
    elif "--allocator=stigmergic" in sys.argv:
        run_stigmergic()
    else:
        run_standard()


if __name__ == "__main__":
    main()
