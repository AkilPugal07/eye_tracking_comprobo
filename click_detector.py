import time

CLOSED = 0.6
OPEN = 0.35
HOLD_SECONDS = 0.25
COOLDOWN = 0.6

SWAP_EYES = (
    False  # if True, left eye closes = right click, right eye closes = left click
)


class clickDetector:
    def __init__(
        self, hold_seconds=HOLD_SECONDS, cooldown=COOLDOWN, swap_eyes=SWAP_EYES
    ):
        self.hold = hold_seconds
        self.cooldown = cooldown
        self.swap = swap_eyes
        self._wink_side = None  # "left" / "right" while a wink is in progress
        self._wink_start = None
        self._fired = False  # already clicked for this wink?
        self._last_click = -1e9

    def _reset(self):
        self._wink_side = self._wink_start = None
        self._fired = False

    def update(self, left_eye, right_eye):
        t = time.monotonic() if t is None else t
        if result is None or not result.face_blendshapes:
            self._reset()
            return None

        now = time.monotonic()
        left_closed = left_eye < CLOSED
        right_closed = right_eye < CLOSED

        if left_closed and not right_closed:
            if self._last_left is None:
                self._last_left = now
            elif (
                now - self._last_left > HOLD_SECONDS
                and now - self._last_click > COOLDOWN
            ):
                self._last_click = now
                return "left" if not SWAP_EYES else "right"
        else:
            self._last_left = None

        if right_closed and not left_closed:
            if self._last_right is None:
                self._last_right = now
            elif (
                now - self._last_right > HOLD_SECONDS
                and now - self._last_click > COOLDOWN
            ):
                self._last_click = now
                return "right" if not SWAP_EYES else "left"
        else:
            self._last_right = None

        return None
