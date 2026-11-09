"""
Detects winks and turns them into clicks.

A wink is when one eye is a lot more closed than the other for HOLD_SECONDS.
blinks close both eyes about the same so they dont click.
we compare the two eyes instead of checking for one closed and one open bc
when you wink the other eye squints too and mediapipe says it's partly closed.

uses mediapipe's eyeBlinkLeft and eyeBlinkRight (0 is open, 1 is closed).
run wink_test.py if the thresholds dont work for you.
"""

import time

import cv2

#numbers from wink_test.py. a wink was about 0.6 for the closed eye and 0.3 for the open one
CLOSED = 0.45 #closed eye has to be above this to start a wink
OPEN = 0.3 #and has to stay above this to keep going
MIN_DIFF = 0.2 #how much more closed the winking eye has to be than the other one
HOLD_SECONDS = 0.25 #normal blinks are ~0.1-0.15s so they wont count
COOLDOWN = 0.6 #min time between clicks

#flip this if left wink does a right click
SWAP_EYES = False


class ClickDetector:
    def __init__(
        self, hold_seconds=HOLD_SECONDS, cooldown=COOLDOWN, swap_eyes=SWAP_EYES,
        closed=CLOSED, open_=OPEN, min_diff=MIN_DIFF,
    ):
        self.hold = hold_seconds
        self.cooldown = cooldown
        self.swap = swap_eyes
        self.closed = closed
        self.open = open_
        self.min_diff = min_diff

        self.left = self.right = None  #latest scores, used for the bars on screen
        self._wink_side = None
        self._wink_start = None
        self._fired = False  #already clicked for this wink
        self._last_click = -1e9

    @property
    def winking(self):
        return self._wink_side

    def _reset(self):
        self._wink_side = self._wink_start = None
        self._fired = False

    def _button(self, eye):
        if self.swap:
            return "right" if eye == "left" else "left"
        return eye

    def update(self, result, t=None):
        """call once per frame. returns "left" or "right" once per wink, otherwise None"""
        t = time.monotonic() if t is None else t
        if result is None or not result.face_blendshapes:
            self.left = self.right = None
            self._reset()
            return None

        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        left, right = scores["eyeBlinkLeft"], scores["eyeBlinkRight"]
        self.left, self.right = left, right

        #wink already going, check if it's still going (looser check than starting one)
        if self._wink_side is not None:
            closed, other = (left, right) if self._wink_side == "left" else (right, left)
            if closed > self.open and closed - other > self.min_diff / 2:
                held = t - self._wink_start
                if (
                    not self._fired
                    and held >= self.hold
                    and t - self._last_click >= self.cooldown
                ):
                    self._fired = True
                    self._last_click = t
                    return self._button(self._wink_side)
                return None
            self._reset()  #eye opened or other eye closed too (blink)

        #new wink starting
        if left > self.closed and left - right > self.min_diff:
            self._wink_side, self._wink_start = "left", t
        elif right > self.closed and right - left > self.min_diff:
            self._wink_side, self._wink_start = "right", t
        return None


def draw_blink_meter(frame, left, right, x=20, y=None, closed=CLOSED, label_left="L", label_right="R"):
    """draws bars for how closed each eye is. the white line is the CLOSED threshold"""
    h = frame.shape[0]
    y = h - 70 if y is None else y
    bar_w, bar_h = 200, 18
    for i, (name, score) in enumerate(((label_left, left), (label_right, right))):
        top = y + i * (bar_h + 10)
        cv2.rectangle(frame, (x + 30, top), (x + 30 + bar_w, top + bar_h), (80, 80, 80), -1)
        if score is not None:
            fill = int(bar_w * min(max(score, 0.0), 1.0))
            color = (0, 0, 255) if score > closed else (0, 200, 0)
            cv2.rectangle(frame, (x + 30, top), (x + 30 + fill, top + bar_h), color, -1)
            cv2.putText(frame, f"{score:.2f}", (x + 40 + bar_w, top + bar_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tick = x + 30 + int(bar_w * closed)
        cv2.line(frame, (tick, top - 3), (tick, top + bar_h + 3), (255, 255, 255), 2)
        cv2.putText(frame, name, (x, top + bar_h - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
