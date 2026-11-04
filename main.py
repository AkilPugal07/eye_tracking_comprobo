"""
Eye-tracking mouse: moves the cursor to where you're looking.

Before the first run (and whenever you sit differently / change setup):
    python calibration.py      # record calibration data
    python gaze_boundary.py    # fit the gaze -> screen model

Then:
    python main.py

Keys (the camera window needs focus -- click it once if keys do nothing):
    SPACE  start / pause mouse control (starts PAUSED)
    q      quit
Emergency stop: push the real mouse into a corner of the screen.

face_landmarker.task must be in this folder:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import cv2
import pyautogui

import mouse_movement
from calibration import FaceTracker  # same camera + detector setup (and flip) as calibration
from gaze_boundary import MODEL_PATH, GazeMapper

SENSITIVITY = 1.0  # >1 stretches eye movement so the screen edges are easier to reach

# Smoothing (One Euro filter, see gaze_boundary.py):
FILTER_MIN_CUTOFF = 0.5  # lower = steadier when looking at one spot, but laggier
FILTER_BETA = 0.5        # higher = follows quick eye jumps faster, but jitterier

# "Sticky" cursor: it stays still until your gaze moves more than this many
# pixels away from it. Your gaze naturally wobbles ~30 px while staring at one
# spot, so this keeps the cursor from creeping around.
MIN_MOVE_PX = 50

GREEN, RED, ORANGE, WHITE = (0, 200, 0), (0, 0, 255), (0, 165, 255), (255, 255, 255)


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("No gaze model found. Run `python calibration.py` and then "
                         "`python gaze_boundary.py` first.")

    mapper = GazeMapper.load(min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA)
    screen_w, screen_h = pyautogui.size()  # real pixels (ignores Windows display scaling)
    screen = mouse_movement.MouseMovement(screen_w, screen_h, SENSITIVITY, MIN_MOVE_PX)
    tracker = FaceTracker()

    controlling = False  # start paused so the mouse isn't grabbed right away
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
            elif not gaze["on_screen"]:
                status, color = "OFF SCREEN - cursor stays put", RED
            else:
                status, color = f"ON SCREEN ({gaze['x']:.2f}, {gaze['y']:.2f})", GREEN
                if controlling:
                    screen.move_to_gaze(gaze["x"], gaze["y"])

            mode = "MOUSE CONTROL ON  (SPACE = pause)" if controlling else "MOUSE PAUSED  (SPACE = start)"
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, mode, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        GREEN if controlling else WHITE, 2)
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
