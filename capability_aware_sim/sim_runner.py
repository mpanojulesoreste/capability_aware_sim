"""Headless simulation engine — no pygame dependency. Used by sweep.py."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from world import USER_HOME, HANDOFF_BASE
from tasks  import make_task_list, Task
from agents import ROBOT_MAX_SPEED, AVOIDANCE_DIST


class _Robot:
    """Minimal robot: straight-line movement + soft separation."""

    def __init__(self, robot_id: int, start_pos):
        self.id    = robot_id
        self.pos   = np.array(start_pos, dtype=float)
        self.vel   = np.zeros(2)
        self.state = "idle"
        self.task  = None
        self.target: Optional[np.ndarray] = None

    def set_target(self, pos):
        self.target = np.array(pos, dtype=float)

    @property
    def at_target(self):
        return self.target is None

    def update(self, dt, others):
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


class _User:
    """Minimal user: position + heading, can reposition."""

    def __init__(self, home_pos, heading_deg=0.0,
                 reach_right=0.8, reach_left=0.25):
        self.pos         = np.array(home_pos, dtype=float)
        self.heading_deg = heading_deg
        self.reach_right = reach_right
        self.reach_left  = reach_left

    def is_reachable(self, pos) -> bool:
        delta = pos - self.pos
        dist  = np.linalg.norm(delta)
        h_rad = np.radians(self.heading_deg)
        right = np.array([np.cos(h_rad - np.pi / 2),
                           np.sin(h_rad - np.pi / 2)])
        reach = self.reach_right if np.dot(delta, right) >= 0 else self.reach_left
        return dist <= reach

    def side(self, pos) -> str:
        """'right' (strong) or 'left' (weak) relative to user heading."""
        delta = pos - self.pos
        h_rad = np.radians(self.heading_deg)
        right = np.array([np.cos(h_rad - np.pi / 2),
                           np.sin(h_rad - np.pi / 2)])
        return "right" if np.dot(delta, right) >= 0 else "left"

    def reachable_boundary_dist(self, pos) -> float:
        """Distance beyond the reachable boundary (0 if inside)."""
        delta = pos - self.pos
        dist  = np.linalg.norm(delta)
        h_rad = np.radians(self.heading_deg)
        right = np.array([np.cos(h_rad - np.pi / 2),
                           np.sin(h_rad - np.pi / 2)])
        reach = self.reach_right if np.dot(delta, right) >= 0 else self.reach_left
        return max(0.0, dist - reach)

    def dist_to(self, pos) -> float:
        return float(np.linalg.norm(pos - self.pos))

    def reposition(self, new_pos):
        self.pos = np.array(new_pos, dtype=float)


class _StigmergicRobot(_Robot):
    """Headless stigmergic robot with seeded random exploration."""

    SENSE_RADIUS  = 2.0
    BEACON_RADIUS = 1.5
    HANDOFF_DIST  = 0.25

    STATE_EXPLORING   = "exploring"
    STATE_COMMITTED   = "committed"
    STATE_CARRYING    = "carrying"
    STATE_APPROACHING = "approaching"
    STATE_HANDING_OFF = "handing_off"

    def __init__(self, robot_id: int, start_pos, rng):
        super().__init__(robot_id, start_pos)
        self.state          = self.STATE_EXPLORING
        self.committed_task = None
        self.carrying_task  = None
        self.in_beacon      = False
        self._rng           = rng
        self._waypoint_idx  = int(rng.integers(0, 5))
        self._explore_target = None
        self.visible_task_ids  = []
        self.visible_robot_ids = []

    def local_tick(self, dt, all_robots, all_tasks, user):
        dist_to_user = np.linalg.norm(self.pos - user.pos)
        self.in_beacon = dist_to_user <= self.BEACON_RADIUS

        self.visible_task_ids = [
            t.task_id for t in all_tasks
            if np.linalg.norm(t.pickup_pos - self.pos) <= self.SENSE_RADIUS
            and t.status not in ("done",)
        ]
        if self.committed_task is not None and \
                self.committed_task.task_id not in self.visible_task_ids:
            if self.committed_task.status not in ("done",):
                self.visible_task_ids.append(self.committed_task.task_id)

        self.visible_robot_ids = [
            r.id for r in all_robots
            if r is not self
            and np.linalg.norm(r.pos - self.pos) <= self.SENSE_RADIUS
        ]

        if self.state == self.STATE_EXPLORING:
            self._rule_explore(dt, all_robots, all_tasks, user)
        elif self.state == self.STATE_COMMITTED:
            self._rule_committed(all_robots, all_tasks, user)
        elif self.state in (self.STATE_CARRYING, self.STATE_APPROACHING):
            self._rule_carry(user)


    def _rule_explore(self, dt, all_robots, all_tasks, user):
        candidate = self._find_best_item(all_robots, all_tasks)
        if candidate is not None:
            self.committed_task = candidate
            candidate.status = "assigned"
            candidate.assigned_robot = self.id
            self.state  = self.STATE_COMMITTED
            self.target = candidate.pickup_pos.copy()
        else:
            self._wander()

    def _rule_committed(self, all_robots, all_tasks, user):
        task = self.committed_task
        if task is None or task.status == "done":
            self._reset_to_explore()
            return
        if self._closer_robot_exists(task, all_robots):
            task.status = "pending"
            task.assigned_robot = None
            self._reset_to_explore()
            return
        if self.at_target:
            self.carrying_task  = task
            self.committed_task = None
            task.status         = "fetching"
            self.state          = self.STATE_CARRYING
            self._set_delivery_target(user)

    def _rule_carry(self, user):
        if self.at_target:
            self.state = self.STATE_APPROACHING
        if self.in_beacon:
            self._set_delivery_target(user)
            self.state = self.STATE_APPROACHING
        dist_to_user = np.linalg.norm(self.pos - user.pos)
        if dist_to_user <= self.HANDOFF_DIST * 3:
            self.state  = self.STATE_HANDING_OFF
            self.target = self.carrying_task.handoff_pos.copy()

    def _find_best_item(self, all_robots, all_tasks):
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
        task = self.carrying_task
        if task is None:
            return
        if self.in_beacon:
            h_rad = np.radians(user.heading_deg)
            right = np.array([np.cos(h_rad - np.pi / 2),
                               np.sin(h_rad - np.pi / 2)])
            handoff = user.pos + right * user.reach_right * 0.88
            task.handoff_pos = handoff
        else:
            task.handoff_pos = user.pos.copy()
        self.target = task.handoff_pos.copy()

    def _wander(self):
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

    def _reset_to_explore(self):
        self.committed_task = None
        self.state  = self.STATE_EXPLORING
        self.target = None


class _MetricsLog:
    def __init__(self, label: str, reach_right: float, reach_left: float):
        self.label         = label
        self.reach_right   = reach_right
        self.reach_left    = reach_left
        self._start_time   = 0.0
        self.completion_times: List[float] = []
        self.reposition_count = 0
        self.unreachable_count = 0
        self.delivery_count    = 0

        self._fluency_dt   = 0.1
        self._fluency_acc  = 0.0
        self._h_idle_ticks = 0
        self._r_idle_ticks = 0
        self._c_act_ticks  = 0
        self._f_del_ticks  = 0
        self._total_ticks  = 0
        self._h_active_until     = -1.0
        self._in_handoff_receive = False

        self.strong_side_count = 0
        self.weak_side_count   = 0
        self.handoff_distances: List[float] = []
        self.handoff_excesses:  List[float] = []

        self.total_robot_distance = 0.0
        self.task_durations:  List[float] = []

    def set_start(self, t: float):
        self._start_time = t

    def _tick_fluency(self, h_active: bool, r_active: bool):
        self._total_ticks += 1
        if not h_active:
            self._h_idle_ticks += 1
        if not r_active:
            self._r_idle_ticks += 1
        if h_active and r_active:
            self._c_act_ticks += 1
        if not h_active and not r_active:
            self._f_del_ticks += 1

    def log_delivery(self, t: float, reachable: bool,
                     handoff_pos, user_pos, user_heading_deg,
                     user_reach_right, user_reach_left,
                     pickup_time: float):
        self.delivery_count += 1
        elapsed = t - self._start_time
        self.completion_times.append(elapsed)
        if not reachable:
            self.unreachable_count += 1

        delta = handoff_pos - user_pos
        h_rad = np.radians(user_heading_deg)
        right = np.array([np.cos(h_rad - np.pi / 2),
                           np.sin(h_rad - np.pi / 2)])
        if np.dot(delta, right) >= 0:
            self.strong_side_count += 1
        else:
            self.weak_side_count += 1

        dist = float(np.linalg.norm(delta))
        self.handoff_distances.append(dist)

        reach = user_reach_right if np.dot(delta, right) >= 0 else user_reach_left
        excess = max(0.0, dist - reach)
        self.handoff_excesses.append(excess)

        if pickup_time >= 0:
            self.task_durations.append(elapsed - (pickup_time - self._start_time))

    def log_reposition(self):
        self.reposition_count += 1

    def total_time(self) -> float:
        return max(self.completion_times) if self.completion_times else 0.0

    def summary(self) -> dict:
        n_strong = self.strong_side_count
        n_weak   = self.weak_side_count
        total_hw = n_strong + n_weak
        strong_frac = n_strong / total_hw if total_hw > 0 else 0.0

        mean_excess = float(np.mean(self.handoff_excesses)) if self.handoff_excesses else 0.0
        max_excess  = float(np.max(self.handoff_excesses))  if self.handoff_excesses else 0.0
        mean_dist   = float(np.mean(self.handoff_distances)) if self.handoff_distances else 0.0

        mean_dur = float(np.mean(self.task_durations))     if self.task_durations else 0.0
        var_dur  = float(np.var(self.task_durations))      if self.task_durations else 0.0

        T = self._total_ticks if self._total_ticks > 0 else 1
        h_idle = self._h_idle_ticks / T
        r_idle = self._r_idle_ticks / T
        c_act  = self._c_act_ticks  / T
        f_del  = self._f_del_ticks  / T

        return {
            "label":                    self.label,
            "total_time":               round(self.total_time(), 3),
            "repositions":              self.reposition_count,
            "unreachable":              self.unreachable_count,
            "deliveries":               self.delivery_count,
            "human_idle_time":          round(h_idle, 4),
            "robot_idle_time":          round(r_idle, 4),
            "concurrent_activity":      round(c_act, 4),
            "functional_delay":         round(f_del, 4),
            "strong_side_handoff_count": n_strong,
            "weak_side_handoff_count":   n_weak,
            "strong_side_fraction":      round(strong_frac, 4),
            "mean_handoff_distance_from_user": round(mean_dist, 4),
            "mean_handoff_excess":       round(mean_excess, 4),
            "max_handoff_excess":        round(max_excess, 4),
            "total_robot_distance":      round(self.total_robot_distance, 4),
            "mean_task_duration":        round(mean_dur, 4),
            "task_completion_variance":  round(var_dur, 4),
        }


def _dist_from_boundary(handoff_pos, user_pos, heading_deg, reach_right, reach_left) -> float:
    """Excess distance beyond the reachable boundary (0 if inside). Pure function."""
    delta = handoff_pos - user_pos
    dist  = np.linalg.norm(delta)
    h_rad = np.radians(heading_deg)
    right = np.array([np.cos(h_rad - np.pi / 2), np.sin(h_rad - np.pi / 2)])
    reach = reach_right if np.dot(delta, right) >= 0 else reach_left
    return float(max(0.0, dist - reach))


def _side(handoff_pos, user_pos, heading_deg) -> str:
    """'right' or 'left' relative to user heading. Pure function."""
    delta = handoff_pos - user_pos
    h_rad = np.radians(heading_deg)
    right = np.array([np.cos(h_rad - np.pi / 2), np.sin(h_rad - np.pi / 2)])
    return "right" if np.dot(delta, right) >= 0 else "left"


def _best_reachable_handoff_headless(default_handoff, user):
    delta = default_handoff - user.pos
    dist  = np.linalg.norm(delta)
    if dist < 1e-6:
        return user.pos + np.array([user.reach_right * 0.9, 0.0])
    direction = delta / dist
    h_rad = np.radians(user.heading_deg)
    right = np.array([np.cos(h_rad - np.pi / 2),
                       np.sin(h_rad - np.pi / 2)])
    on_right = np.dot(delta, right) >= 0
    reach = user.reach_right if on_right else user.reach_left
    if dist <= reach:
        return default_handoff.copy()
    return user.pos + direction * reach * 0.95


def _handoff_penalty_headless(handoff_pos, user):
    delta = handoff_pos - user.pos
    dist  = np.linalg.norm(delta)
    h_rad = np.radians(user.heading_deg)
    right = np.array([np.cos(h_rad - np.pi / 2),
                       np.sin(h_rad - np.pi / 2)])
    on_right = np.dot(delta, right) >= 0
    reach = user.reach_right if on_right else user.reach_left
    excess = max(0.0, dist - reach)
    side_penalty = 0.0 if on_right else 0.4
    return excess + side_penalty


def _robot_travel_cost(robot, task):
    to_pickup  = np.linalg.norm(task.pickup_pos - robot.pos)
    to_handoff = np.linalg.norm(task.handoff_pos - task.pickup_pos)
    return to_pickup + to_handoff


def _baseline_alloc(robots, tasks, user):
    pending = [t for t in tasks if t.status == "pending"]
    idle    = [r for r in robots if r.state == "idle"]
    if not pending or not idle:
        return None
    best, best_cost = None, float("inf")
    for robot in idle:
        for task in pending:
            cost = _robot_travel_cost(robot, task)
            if cost < best_cost:
                best_cost = cost
                best = (robot, task, task.handoff_pos.copy())
    return best


def _adaptive_alloc(robots, tasks, user):
    W_TRAVEL = 1.0
    W_HANDOFF = 2.5
    pending = [t for t in tasks if t.status == "pending"]
    idle    = [r for r in robots if r.state == "idle"]
    if not pending or not idle:
        return None
    best, best_cost, best_handoff = None, float("inf"), None
    for robot in idle:
        for task in pending:
            handoff   = _best_reachable_handoff_headless(task.handoff_pos, user)
            t_cost    = _robot_travel_cost(robot, task)
            h_penalty = _handoff_penalty_headless(handoff, user)
            cost      = W_TRAVEL * t_cost + W_HANDOFF * h_penalty
            if cost < best_cost:
                best_cost = cost
                best = (robot, task, handoff)
                best_handoff = handoff
    return best


FLUENCY_SAMPLE_DT = 0.1


def run_headless(allocator_name: str,
                 reach_right: float = 0.8,
                 reach_left:  float = 0.25,
                 seed:        int   = 42,
                 dt:          float = 0.05,
                 profile_name: str  = "baseline_test") -> tuple:
    """
    Run a complete 5-task scenario headlessly.

    Returns
    -------
    (metrics_summary: dict, per_task_records: list[dict])
    """
    user   = _User(USER_HOME, heading_deg=0.0,
                   reach_right=reach_right, reach_left=reach_left)
    starts = [(0.4, 0.4), (0.4, 3.6), (2.5, 2.0)]
    tasks  = make_task_list()

    is_stigmergic = (allocator_name == "stigmergic")

    if is_stigmergic:
        master_rng  = np.random.default_rng(seed)
        robot_seeds = master_rng.integers(0, 2**31, size=3).tolist()
        robots = [
            _StigmergicRobot(i, starts[i],
                             rng=np.random.default_rng(robot_seeds[i]))
            for i in range(3)
        ]
    else:
        robots = [_Robot(i, starts[i]) for i in range(3)]

    metrics = _MetricsLog(allocator_name, reach_right, reach_left)

    alloc_fn = {
        "baseline":    _baseline_alloc,
        "adaptive":    _adaptive_alloc,
        "stigmergic":  None,
    }[allocator_name]

    sim_time       = 0.0
    started        = False
    done           = False
    per_task: dict = {}
    prev_pos       = [r.pos.copy() for r in robots]
    fluency_acc    = 0.0
    h_active_until = -1.0
    max_iters      = int(300 / dt)

    for _ in range(max_iters):
        if done:
            break

        sim_time += dt
        if not started:
            metrics.set_start(sim_time)
            started = True

        for robot in robots:
            robot.update(dt, robots)

        for i, robot in enumerate(robots):
            d = float(np.linalg.norm(robot.pos - prev_pos[i]))
            metrics.total_robot_distance += d
            prev_pos[i] = robot.pos.copy()

        if is_stigmergic:
            for robot in robots:
                robot.local_tick(dt, robots, tasks, user)

        for robot in robots:
            if not robot.at_target:
                continue

            if is_stigmergic:
                task = robot.carrying_task
                if task is not None and robot.state in (
                        _StigmergicRobot.STATE_HANDING_OFF,
                        _StigmergicRobot.STATE_APPROACHING):
                    reachable = user.is_reachable(task.handoff_pos)
                    # snapshot user position BEFORE any reposition so all
                    # excess/boundary metrics reflect the actual delivery geometry
                    user_pos_at_delivery = user.pos.copy()
                    dist_from_boundary   = float(_dist_from_boundary(
                        task.handoff_pos, user_pos_at_delivery,
                        user.heading_deg, user.reach_right, user.reach_left))
                    if not reachable:
                        nudge = task.handoff_pos - np.array([0.25, 0.0])
                        user.reposition(nudge)
                        metrics.log_reposition()
                    task.status    = "done"
                    task.done_time = sim_time

                    rec = per_task.get(task.task_id, {})
                    rec.update({
                        "task_id":            task.task_id,
                        "robot_id":           robot.id,
                        "pickup_time":        float(task.start_time or sim_time),
                        "dropoff_time":       float(sim_time),
                        "handoff_pos":        task.handoff_pos.tolist(),
                        "reachable":          bool(reachable),
                        "side":               _side(task.handoff_pos, user_pos_at_delivery, user.heading_deg),
                        "dist_from_boundary": dist_from_boundary,
                    })
                    per_task[task.task_id] = rec

                    metrics.log_delivery(
                        sim_time, reachable,
                        task.handoff_pos, user_pos_at_delivery, user.heading_deg,
                        user.reach_right, user.reach_left,
                        task.start_time or sim_time,
                    )
                    h_active_until      = sim_time + 1.5
                    robot.carrying_task = None
                    robot.state         = _StigmergicRobot.STATE_EXPLORING
                    robot.target        = None

            else:
                if robot.state == "fetching":
                    robot.state = "delivering"
                    robot.set_target(robot.task.handoff_pos)
                    robot.task.status = "delivering"

                elif robot.state == "delivering":
                    task      = robot.task
                    reachable = user.is_reachable(task.handoff_pos)
                    # snapshot user position BEFORE any reposition so all
                    # excess/boundary metrics reflect the actual delivery geometry
                    user_pos_at_delivery = user.pos.copy()
                    dist_from_boundary   = float(_dist_from_boundary(
                        task.handoff_pos, user_pos_at_delivery,
                        user.heading_deg, user.reach_right, user.reach_left))
                    if not reachable:
                        nudge = task.handoff_pos - np.array([0.25, 0.0])
                        user.reposition(nudge)
                        metrics.log_reposition()

                    task.status    = "done"
                    task.done_time = sim_time

                    rec = per_task.get(task.task_id, {})
                    rec.update({
                        "task_id":            task.task_id,
                        "robot_id":           robot.id,
                        "pickup_time":        float(task.start_time or sim_time),
                        "dropoff_time":       float(sim_time),
                        "handoff_pos":        task.handoff_pos.tolist(),
                        "reachable":          bool(reachable),
                        "side":               _side(task.handoff_pos, user_pos_at_delivery, user.heading_deg),
                        "dist_from_boundary": dist_from_boundary,
                    })
                    per_task[task.task_id] = rec

                    metrics.log_delivery(
                        sim_time, reachable,
                        task.handoff_pos, user_pos_at_delivery, user.heading_deg,
                        user.reach_right, user.reach_left,
                        task.start_time or sim_time,
                    )
                    h_active_until = sim_time + 1.5

                    robot.state = "idle"
                    robot.task  = None

        if not is_stigmergic:
            result = alloc_fn(robots, tasks, user)
            if result is not None:
                robot, task, handoff   = result
                task.handoff_pos       = handoff
                task.status            = "assigned"
                task.start_time        = sim_time
                task.assigned_robot    = robot.id
                robot.task             = task
                robot.state            = "fetching"
                robot.set_target(task.pickup_pos)
                per_task[task.task_id] = {"task_id": task.task_id}

        fluency_acc += dt
        while fluency_acc >= FLUENCY_SAMPLE_DT:
            fluency_acc -= FLUENCY_SAMPLE_DT
            h_active = sim_time <= h_active_until
            r_active = any(r.state != "idle" for r in robots)
            metrics._tick_fluency(h_active, r_active)

        if all(t.status == "done" for t in tasks):
            done = True

    summary = metrics.summary()

    records = []
    for tid in sorted(per_task.keys()):
        rec = per_task[tid]
        rec["profile"]   = profile_name
        rec["allocator"] = allocator_name
        rec["seed"]      = seed
        records.append(rec)

    return summary, records
