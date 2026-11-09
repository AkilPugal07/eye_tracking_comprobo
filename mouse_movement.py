#handles moving the mouse and clicking
import math
import time

import pyautogui

#pyautogui waits 0.1s after every call by default which slows the loop down to ~10 fps
pyautogui.PAUSE = 0


class MouseMovement:
    """
    A class to handle mouse movement based on screen dimensions and sensitivity."""

    def __init__(
        self,
        screen_width,
        screen_height,
        sensitivity=1.0,
        min_move_px=0,
        smoothing=0.2,
        max_speed=1200,
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sensitivity = sensitivity
        self.min_move_px = min_move_px  #ignore movements smaller than this
        self.smoothing = smoothing  #0 means no easing, jumps straight there
        self.max_speed = max_speed  #None means no speed limit
        self._last_pos = None  #last position sent to the mouse
        self._cx = None  #smoothed cursor position
        self._cy = None
        self._t = None

    def _clamp(self, x, y):
        x = max((1, min(self.screen_width - 2, x)))
        y = max((1, min(self.screen_height - 2, y)))
        return x, y

    def _move_clamped(self, x, y):
        """Move to pixel (x, y), kept inside the screen."""
        #stays 1px from the edges bc pyautogui fail-safe stops the program if mouse hits a corner
        x, y = self._clamp(x, y)
        ix, iy = int(round(x)), int(round(y))
        if self._last_pos != (ix, iy):
            pyautogui.moveTo(ix, iy)
            self._last_pos = (ix, iy)
        self._cx, self._cy = x, y

    def _ensure_started(self):
        #start from where the cursor actually is
        if self._cx is None:
            self._cx, self._cy = pyautogui.position()

    def move_mouse(self, x, y):
        """
        Move the cursor to specified location (x,y) on the screen
        """
        # Scale the coordinates based on sensitivity
        self._move_clamped(x * self.sensitivity, y * self.sensitivity)

    def move_to_gaze(self, gaze_x, gaze_y, t=None):
        """
        Move the cursor to where the user is looking.

        gaze_x and gaze_y are fractions of the screen from GazeMapper (0 to 1 is on screen).
        sensitivity >1 makes the edges easier to reach
        """
        t = time.monotonic() if t is None else t
        self._ensure_started()
        if self._t is None:
            self._t = t
            return
        dt = min(max(t - self._t, 0.0), 0.1)  #cap dt so theres no huge jump after a pause
        self._t = t

        fx = 0.5 + (gaze_x - 0.5) * self.sensitivity
        fy = 0.5 + (gaze_y - 0.5) * self.sensitivity
        tx, ty = self._clamp(fx * self.screen_width, fy * self.screen_height)

        dx, dy = tx - self._cx, ty - self._cy
        dist = math.hypot(dx, dy)

        # Filters out noise by ignoring small movements and smoothing the cursor movement
        if self.min_move_px > 0 and dist < self.min_move_px:
            return

        if self.smoothing > 0:
            alpha = 1.0 - math.exp(-dt / self.smoothing)
            dx, dy = dx * alpha, dy * alpha

        if self.max_speed:
            step = math.hypot(dx, dy)
            max_step = self.max_speed * dt
            if step > max_step:
                dx, dy = dx * max_step / step, dy * max_step / step

        self._move_clamped(self._cx + dx, self._cy + dy)

    def pause(self):
        """
        Pause the mouse movement, resetting the internal time to avoid jumps when resuming.
        """
        self._t = None

    def right_click(self, has_closed_right_eye):
        """
        Perform a right-click if the right eye is closed.
        """
        if has_closed_right_eye:
            pyautogui.click(button="right")

    def left_click(self, has_closed_left_eye):
        """
        Perform a left-click if the left eye is closed.
        """
        if has_closed_left_eye:
            pyautogui.click(button="left")
