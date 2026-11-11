"""
Eye-tracking mouse: moves the cursor to where you're looking.

Run before the first run and when pose/environment changes a lot:
    python calibration.py #records calibration data

Then:
    python main.py
main.py always uses the newest complete calibration: if there's one newer than
the saved gaze model, it refits the model automatically at startup.

Keys:
    SPACE:start / pause mouse control
    q:quit
Clicking: wink and hold ~0.25 s.
    left eye = left click, right eye = right click (see click_detector.py)
Emergency stop: push the real mouse into a corner of the screen.

face_landmarker.task must be in this folder:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import time

import cv2
import pyautogui

import mouse_movement
from click_detector import ClickDetector, draw_blink_meter
from calibration import (
    FaceTracker,
)  # same camera + detector setup as calibration
from gaze_boundary import GazeMapper, ensure_model_is_current

SENSITIVITY = 1.0  # >1 stretches eye movement so the screen edges are easier to reach

# Smoothing
FILTER_MIN_CUTOFF = 0.5  # lower means steadier when looking at one spot, but laggier
FILTER_BETA = 0.5  # higher follows quick eye jumps faster, but jitterier

# cursor only moves if gaze shifts by more than these amounts of pixels
MIN_MOVE_PX = 25

GREEN, RED, ORANGE, WHITE = (0, 200, 0), (0, 0, 255), (0, 165, 255), (255, 255, 255)

SMOOTHING = 0.2  # seconds for the One Euro filter to smooth the cursor movement
MAX_SPEED = 900  # px/sec, to avoid huge jumps if the gaze model fails for a frame


def main():
    # use the newest complete calibration
    ensure_model_is_current()

    mapper = GazeMapper.load(min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA)
    screen_w, screen_h = pyautogui.size()
    screen = mouse_movement.MouseMovement(
        screen_w,
        screen_h,
        SENSITIVITY,
        MIN_MOVE_PX,
        smoothing=SMOOTHING,
        max_speed=MAX_SPEED,
    )
    clicker = ClickDetector()
    tracker = FaceTracker()

    controlling = False  # start paused
    click_text, click_text_until = "", 0.0
    print(f"Screen {screen_w}x{screen_h}. SPACE = start/pause mouse control, q = quit.")

    try:
        while True:
            frame, result = tracker.read()
            if frame is None:
                print("No frame from camera.")
                break

            gaze = mapper.update(result)

            if gaze is None:
                status, color = "no face / blinking", ORANGE
                screen.pause()  # reset time so the next gaze jump isn't huge
            elif not gaze["on_screen"]:
                status, color = "OFF SCREEN - cursor stays put", RED
                screen.pause()  #
            else:
                status, color = f"ON SCREEN ({gaze['x']:.2f}, {gaze['y']:.2f})", GREEN
                if controlling:
                    screen.move_to_gaze(gaze["x"], gaze["y"])
                else:
                    screen.pause()
            # clicks when wink detected. gaze does not move when eye is detected to be closed
            button = clicker.update(result)
            if button is not None:
                if controlling:
                    if button == "left":
                        screen.left_click(True)
                    else:
                        screen.right_click(True)
                    click_text = f"{button.upper()} CLICK"
                else:
                    click_text = f"{button.upper()} WINK (paused - no click)"
                click_text_until = time.monotonic() + 0.8
            if time.monotonic() < click_text_until:
                cv2.putText(
                    frame,
                    click_text,
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    ORANGE,
                    2,
                )
            elif clicker.winking:
                cv2.putText(
                    frame,
                    f"{clicker.winking} wink...",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    WHITE,
                    2,
                )
            # live eye-closure scores. red means eye is closed
            eye_l, eye_r = (
                (clicker.right, clicker.left)
                if clicker.swap
                else (clicker.left, clicker.right)
            )
            draw_blink_meter(frame, eye_l, eye_r, closed=clicker.closed)

            mode = (
                "MOUSE CONTROL ON  (SPACE = pause)"
                if controlling
                else "MOUSE PAUSED  (SPACE = start)"
            )
            cv2.putText(
                frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )
            cv2.putText(
                frame,
                mode,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                GREEN if controlling else WHITE,
                2,
            )
            cv2.imshow("Eye mouse", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                controlling = not controlling
    except pyautogui.FailSafeException:
        print("Fail-safe triggered (mouse pushed into a screen corner) -- stopped.")
    finally:
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
