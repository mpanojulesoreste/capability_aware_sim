"""Room bounds and key positions (all in meters)."""

ROOM_W = 5.0   # meters
ROOM_H = 4.0   # meters

PIXELS_PER_METER = 100
PANEL_W = int(ROOM_W * PIXELS_PER_METER)   # 500 px
PANEL_H = int(ROOM_H * PIXELS_PER_METER)   # 400 px
SIDEBAR_W = 220                             # px, live metrics column

# User home position (meters)
USER_HOME = (1.0, 2.0)

# Item spawn positions (meters) — used by task script
ITEM_POSITIONS = [
    (4.2, 0.5),
    (4.5, 3.5),
    (2.5, 0.3),
    (3.8, 2.0),
    (1.8, 3.6),
]

# Handoff target (meters) — where robots deliver next to user
HANDOFF_BASE = (1.6, 2.0)
