"""Task dataclass and scripted task list."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from world import ITEM_POSITIONS, HANDOFF_BASE


@dataclass
class Task:
    task_id:       int
    pickup_pos:    np.ndarray          # meters
    handoff_pos:   np.ndarray          # meters (allocator may override)
    status:        str = "pending"     # pending | assigned | fetching | delivering | done
    assigned_robot: Optional[int] = None
    start_time:    Optional[float] = None
    done_time:     Optional[float] = None

    # Colors per status (for rendering)
    STATUS_COLORS = {
        "pending":    (220, 220,  60),
        "assigned":   (200, 140,  40),
        "fetching":   ( 80, 180, 230),
        "delivering": (180,  80, 230),
        "done":       ( 80, 210, 100),
    }

    @property
    def color(self):
        return self.STATUS_COLORS.get(self.status, (180, 180, 180))


def make_task_list() -> list[Task]:
    """Return the canonical scripted task list (same for both panels)."""
    offsets = [
        (0.0,  0.0),
        (0.1, -0.05),
        (-0.1, 0.05),
        (0.0,  0.1),
        (-0.05, -0.1),
    ]
    tasks = []
    for i, (item_pos, offset) in enumerate(zip(ITEM_POSITIONS, offsets)):
        handoff = (HANDOFF_BASE[0] + offset[0],
                   HANDOFF_BASE[1] + offset[1])
        tasks.append(Task(
            task_id=i,
            pickup_pos=np.array(item_pos, dtype=float),
            handoff_pos=np.array(handoff, dtype=float),
        ))
    return tasks
