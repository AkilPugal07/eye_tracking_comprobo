# A library to handle mouse movement based on screen dimensions and sensitivity.
import pyautogui


class MouseMovement:
    """
    A class to handle mouse movement based on screen dimensions and sensitivity."""

    def __init__(self, screen_width, screen_height, sensitivity=1.0):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sensitivity = sensitivity

    def move_mouse(self, x, y):
        """
        Move the cursor to specified location (x,y) on the screen
        """
        # Scale the coordinates based on sensitivity
        scaled_x = int(x * self.sensitivity)
        scaled_y = int(y * self.sensitivity)

        # Ensure the mouse doesn't go out of screen bounds
        scaled_x = max(0, min(self.screen_width - 1, scaled_x))
        scaled_y = max(0, min(self.screen_height - 1, scaled_y))

        pyautogui.moveTo(scaled_x, scaled_y)

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
