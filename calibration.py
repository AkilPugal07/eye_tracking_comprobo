"""
Gaze calibration data collection file

Shows 9 targets on the screen followed by 8 targets around the
screen. For every target the user gets a short moment to move their eyes
there, then face data is recorded for RECORD_SECONDS and saved to a file.
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk

# import csv
# import json
# from datetime import datetime


MODEL_PATH = Path(__file__).with_name("face_landmarker.task")
OUTPUT_DIR = Path(__file__).with_name("calibration_data")

RECORD_SECONDS = 3.0

WINDOW = "Calibration"


def build_targets():
    """Return a list of 17 targets (9 on-screen, then 8 off-screen).

    Each target in dictionary:
        {"label": "top_left", "x": 0.1, "y": 0.1, "on_screen": True}

    using normalized coordinates, where (0, 0) is the top-left of the
    screen and (1, 1) is the bottom-right. Converting to pixels by
    multiplying by the screen width/height.
    """
    #9 on screen points
    targets = []
    for y in [0.1, 0.5, 0.9]:
        for x in [0.1, 0.5, 0.9]:
            targets.append({"label": f"on_screen_{x}_{y}", "x": x, "y": y, "on_screen": True})
    #8 off-screen points
    for y in [-0.2, 1.2]:
        for x in [0, 0.5, 1]:
            targets.append({"label": f"off_screen_{x}_{y}", "x": x, "y": y, "on_screen": False})
    for x in [-0.2, 1.2]:
        for y in [0, 0.5, 1]:
            targets.append({"label": f"off_screen_{x}_{y}", "x": x, "y": y, "on_screen": False})
    return targets


def get_screen_size():
    """Return (width, height) of the screen in pixels.
    using tkinter: Tk().winfo_screenwidth() and winfo_screenheight()
    """
    root = tk.Tk()
    root.withdraw()#hide the root window
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    return width, height



class FaceTracker:
    """add descritpion"""

    def __init__(self, camera_index=0):
        # TODO: open the camera (see main.py)
        self.cap = cv2.VideoCapture(camera_index)
        # TODO: create the FaceLandmarker (see main.py), and consider:
        #   - output_facial_transformation_matrixes=True  -> head pose, useful for gaze
        #   - running_mode=vision.RunningMode.VIDEO       -> faster tracking between
        #     frames, but then you must call detect_for_video(image, timestamp_ms)
        #     with a timestamp that always increases
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def read(self):
        """Grab one camera frame, run detection, and return the result.

        main.py flips the image -- should calibration do the same so the data
        matches what your tracking code will see later?
        """
        # TODO
        

    def close(self):
        # TODO: release the camera
        raise NotImplementedError


def draw_on_screen_target(canvas, px, py, progress):
    """Draw a dot at pixel (px, py).

    `progress` goes 0 -> 1 over the target's time. Could you use it to show
    the user how long is left (e.g. a shrinking ring)?
    """
    # TODO
    raise NotImplementedError


def draw_off_screen_target(canvas, target, progress):
    """You can't draw a dot off the screen! How will you tell the user where
    to look?

    Ideas: an arrow pointing toward the target + text like
    "look past the top-left corner". Sticky notes on the wall also work.
    """
    # TODO
    raise NotImplementedError



def extract_features(result):
    """Turn one MediaPipe result into plain numbers you can save.

    Return None if no face was detected.

    Things available on `result` (print them to explore):
      - result.face_landmarks[0]                 -> landmarks with .x .y .z
      - result.face_blendshapes[0]               -> items with .category_name .score
                                                    (look for eyeLookUp/Down/In/Out!)
      - result.facial_transformation_matrixes[0] -> 4x4 head pose matrix

    Decide: save ALL landmarks, or only the eye/iris ones?
    (Saving everything is more flexible; you can always drop data later.)
    """
    # TODO
    raise NotImplementedError


def run_target(tracker, screen_size, target, rows):
    """Show one target for RECORD_SECONDS, appending one row per frame to `rows`.

    Rough shape:
        start = time.monotonic()
        while (time since start) < duration:
            read a frame + detect
            if recording: rows.append({...target info..., ...features...})
            draw the canvas, imshow, waitKey(1)

    Each row should include WHICH target was showing (label, x, y, on_screen)
    -- that's the "answer" your model will learn to predict.

    Return False if the user pressed ESC (key code 27) so main() can stop.
    """
    # TODO
    raise NotImplementedError


def save(rows):
    """Write the recorded rows to a file in OUTPUT_DIR.

    Decisions:
      - Format? CSV (csv.writer; opens in a spreadsheet, one column per number)
        or something like np.savez / JSON.
      - Filename: include the date/time so runs don't overwrite each other.
      - Frames where no face was found: skip them, or keep them with a
        face_detected = 0 column?
      - Also worth saving: screen size, camera resolution, your settings.

    Remember OUTPUT_DIR.mkdir(exist_ok=True) before writing.
    """
    # TODO
    raise NotImplementedError

def main():
    # TODO: get screen size, build targets, create FaceTracker
    # TODO: create a fullscreen window
    # TODO: (optional) show an instructions screen, wait for SPACE (key code 32)
    # TODO: loop over targets, call run_target; stop early on ESC
    #       (optional: pause screen between the on-screen and off-screen parts)
    # TODO: close the tracker + cv2.destroyAllWindows()  (try/finally is handy)
    # TODO: save the rows
    raise NotImplementedError


if __name__ == "__main__":
    main()
