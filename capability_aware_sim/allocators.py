"""
Baseline and adaptive allocators.

Each allocator takes the current robot list, pending task list, and user agent,
and returns a (robot, task, handoff_pos) triple — or None if no assignment
is possible right now.
"""

import numpy as np
from agents import REACH_RIGHT, REACH_LEFT

W_TRAVEL  = 1.0
W_HANDOFF = 2.5


def _robot_travel_cost(robot, task):
    """Round-trip estimate: pickup → handoff."""
    to_pickup  = np.linalg.norm(task.pickup_pos - robot.pos)
    to_handoff = np.linalg.norm(task.handoff_pos - task.pickup_pos)
    return to_pickup + to_handoff


def baseline_allocator(robots, tasks, user):
    """
    Capability-blind: pick (robot, task) that minimises robot travel cost.
    Handoff point is whatever the task's default handoff_pos is — no
    adjustment for user reach. Does NOT read user.reach_left/reach_right.
    Returns (robot, task, handoff_pos) or None.
    """
    pending  = [t for t in tasks if t.status == "pending"]
    idle     = [r for r in robots if r.state == "idle"]
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


def _handoff_penalty(handoff_pos, user):
    """
    Distance from handoff_pos to the nearest reachable point on user's
    asymmetric workspace.  Uses right-side vs left-side reach radii.
    """
    delta = handoff_pos - user.pos
    dist  = np.linalg.norm(delta)

    h_rad  = np.radians(user.heading_deg)
    right  = np.array([np.cos(h_rad - np.pi / 2),
                        np.sin(h_rad - np.pi / 2)])
    on_right = np.dot(delta, right) >= 0

    reach = REACH_RIGHT if on_right else REACH_LEFT
    excess = max(0.0, dist - reach)
    side_penalty = 0.0 if on_right else 0.4
    return excess + side_penalty


def _best_reachable_handoff(default_handoff, user):
    """
    If default_handoff is outside reach, nudge it to the nearest reachable
    point on the stronger (right) side, along the user→handoff ray.
    """
    delta = default_handoff - user.pos
    dist  = np.linalg.norm(delta)
    if dist < 1e-6:
        return user.pos + np.array([REACH_RIGHT * 0.9, 0.0])

    direction = delta / dist

    h_rad = np.radians(user.heading_deg)
    right = np.array([np.cos(h_rad - np.pi / 2),
                       np.sin(h_rad - np.pi / 2)])
    on_right = np.dot(delta, right) >= 0
    reach = REACH_RIGHT if on_right else REACH_LEFT

    if dist <= reach:
        return default_handoff.copy()

    return user.pos + direction * reach * 0.95


def adaptive_allocator(robots, tasks, user):
    """
    Capability-aware: minimise weighted sum of robot travel cost and
    handoff penalty.  Adjusts handoff point toward the user's reachable zone.
    Returns (robot, task, handoff_pos) or None.
    """
    pending  = [t for t in tasks if t.status == "pending"]
    idle     = [r for r in robots if r.state == "idle"]
    if not pending or not idle:
        return None

    best, best_cost, best_handoff = None, float("inf"), None
    for robot in idle:
        for task in pending:
            handoff   = _best_reachable_handoff(task.handoff_pos, user)
            t_cost    = _robot_travel_cost(robot, task)
            h_penalty = _handoff_penalty(handoff, user)
            cost      = W_TRAVEL * t_cost + W_HANDOFF * h_penalty
            if cost < best_cost:
                best_cost   = cost
                best        = (robot, task, handoff)
                best_handoff = handoff

    return best
