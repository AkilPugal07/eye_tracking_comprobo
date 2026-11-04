# A library to handle mouse movement based on screen dimensions and sensitivity.
import math

import pyautogui

# By default pyautogui sleeps 0.1 s after EVERY call. We move the mouse once per
# camera frame, so that would cap the whole loop at ~10 fps. Turn it off.
pyautogui.PAUSE = 0


class MouseMovement:
    """
    A class to handle mouse movement based on screen dimensions and sensitivity."""

    def __init__(self, screen_width, screen_height, sensitivity=1.0, min_move_px=0):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sensitivity = sensitivity
        self.min_move_px = min_move_px  # ignore jumps smaller than this (less jitter)
        self._last_pos = None

    def _move_clamped(self, x, y):
        """Move to pixel (x, y), kept inside the screen."""
        # Stay 1 px away from the edges: pyautogui's fail-safe stops the program
        # (FailSafeException) if the mouse lands exactly in a screen corner.
        x = int(max(1, min(self.screen_width - 2, x)))
        y = int(max(1, min(self.screen_height - 2, y)))
        pyautogui.moveTo(x, y)
        self._last_pos = (x, y)

    def move_mouse(self, x, y):
        """
        Move the cursor to specified location (x,y) on the screen
        """
        # Scale the coordinates based on sensitivity
        self._move_clamped(x * self.sensitivity, y * self.sensitivity)

    def move_to_gaze(self, gaze_x, gaze_y):
        """
        Move the cursor to where the user is looking.

        gaze_x, gaze_y are screen fractions from gaze_boundary.GazeMapper
        (0..1 = on screen). Sensitivity stretches eye movement around the screen
        center: 1.0 = exact, >1 makes the screen edges easier to reach.
        """
        fx = 0.5 + (gaze_x - 0.5) * self.sensitivity
        fy = 0.5 + (gaze_y - 0.5) * self.sensitivity
        x = fx * self.screen_width
        y = fy * self.screen_height

        if self._last_pos is not None and self.min_move_px > 0:
            if math.hypot(x - self._last_pos[0], y - self._last_pos[1]) < self.min_move_px:
                return  # tiny change, probably just noise -- keep the cursor still

        self._move_clamped(x, y)

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
