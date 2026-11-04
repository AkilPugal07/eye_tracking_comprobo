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

import csv
import json
from datetime import datetime


MODEL_PATH = Path(__file__).with_name("face_landmarker.task")
OUTPUT_DIR = Path(__file__).with_name("calibration_data")

RECORD_SECONDS = 3.0  # data is recorded for this long; the ring shrinks over it

# Time before recording starts, so the eyes can get to the target first.
# Frames from this period are NOT saved (eyes are still moving).
SETTLE_SECONDS = 1.0             # on-screen: move eyes to the dot
OFF_SCREEN_SETTLE_SECONDS = 2.5  # off-screen: read the arrow + look away

# Head-movement part: look at a dot while slowly moving the head. This teaches
# gaze_boundary.py how much head rotation changes the eye readings, so the
# cursor stays put when you move your head.
HEAD_MOVE_SECONDS = 6.0
HEAD_MOVE_POINTS = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]

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
    off_points = [
        (-0.2, -0.2), (0.5, -0.2), (1.2, -0.2),
        (-0.2, 0.5),               (1.2, 0.5),
        (-0.2, 1.2),  (0.5, 1.2),  (1.2, 1.2),
    ]
    for x, y in off_points:
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
    root.destroy()  # close the hidden Tk window
    return width, height



class FaceTracker:
    """add descritpion"""

    def __init__(self, camera_index=0):
        #open camera
        self.cap = cv2.VideoCapture(camera_index)
        # create the FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def read(self):
        """Grab one camera frame, run detection, and return the result.

        main.py flips the image -- should calibration do the same so the data
        matches what your tracking code will see later?
        """
        success, frame = self.cap.read() #grabs a frame from camera
        if not success:
            return None, None

        frame = cv2.flip(frame, 1)#flip horizontally so it matches main.py

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)#mediapipe needs RBG wrapping in mp.ImageFormat

        result = self.detector.detect(mp_image)#face landmark detection

        return frame, result


    def close(self):
        self.cap.release()
        self.detector.close()

def draw_on_screen_target(canvas, px, py, progress, ring_color=(0, 0, 255)):
    """Draw a dot at pixel (px, py).

    `progress` goes from 0 to 1 over the target's time and a ring around the dot shrinks as time passes.
    `ring_color` lets the caller show a different color before recording starts.
    """
    cv2.circle(canvas, (int(px),int(py)), 8, (0,0,255), -1)
    ring_rad=int(60-(60-8)*progress)
    cv2.circle(canvas, (int(px),int(py)), ring_rad, ring_color, 3)

def draw_off_screen_target(canvas, target, progress):
    """draws an arrow and description for where the user should look for this calibration steps.
    """
    h, w = canvas.shape[:2]# screen size comes from the canvas itself
    margin = 50# keep the arrowhead from being cut off at the edge

    # arrow starts in the middle of the screen
    center = (w//2, h//2)

    end_x = int(min(max(target["x"]*w, margin), w-margin))
    end_y = int(min(max(target["y"]*h, margin), h-margin))
    cv2.arrowedLine(canvas, center, (end_x, end_y), (0, 0, 255), 5, tipLength=0.05)#draws arrow

    text = "Look about 1 inch past the arrow tip and hold until the beep"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 1.0, 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    text_x = w// 2-text_w // 2
    text_y = h//2 - 60 if target["y"] > 1 else h //2+80
    cv2.putText(canvas, text, (text_x, text_y), font, scale, (255, 255, 255), thickness)



def extract_features(result):
    """Turn one MediaPipe result into plain numbers you can save.

    Return None if no face was detected.

    Things available on `result` (print them to explore):
      - result.face_landmarks[0]                 -> landmarks with .x .y .z
      - result.face_blendshapes[0]               -> items with .category_name .score
                                                    (look for eyeLookUp/Down/In/Out!)
      - result.facial_transformation_matrixes[0] -> 4x4 head pose matrix

    Saves everything (flexible -- columns can be dropped later):
      lm{i}_x / lm{i}_y / lm{i}_z  -> all 478 landmarks (468-477 are the irises)
      bs_{name}                    -> 52 blendshape scores
      head_m{row}{col}             -> 16 values of the 4x4 head pose matrix
    """
    if result is None or not result.face_landmarks:
        return None

    features = {}

    for i, lm in enumerate(result.face_landmarks[0]):
        features[f"lm{i}_x"] = lm.x
        features[f"lm{i}_y"] = lm.y
        features[f"lm{i}_z"] = lm.z

    if result.face_blendshapes:
        for shape in result.face_blendshapes[0]:
            features[f"bs_{shape.category_name}"] = shape.score

    if result.facial_transformation_matrixes:
        matrix = np.asarray(result.facial_transformation_matrixes[0])
        for r in range(4):
            for c in range(4):
                features[f"head_m{r}{c}"] = float(matrix[r, c])

    return features


def beep():
    """Short sound so the user knows an off-screen target is done
    (they're looking away, so they can't see the screen)."""
    try:
        import winsound  # Windows only
        winsound.Beep(1000, 200)  # 1000 Hz for 200 ms
    except (ImportError, RuntimeError):
        print("\a", end="", flush=True)  # terminal bell on other systems


def show_message(screen_size, lines):
    """Show lines of centered text and wait for a key.

    Returns True on SPACE, False on ESC (or if the window is closed).
    """
    w, h = screen_size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = h // 2 - len(lines) * 25
    for line in lines:
        (text_w, _), _ = cv2.getTextSize(line, font, 1.0, 2)
        cv2.putText(canvas, line, (w // 2 - text_w // 2, y), font, 1.0, (255, 255, 255), 2)
        y += 50
    cv2.imshow(WINDOW, canvas)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 32:  # SPACE
            return True
        if key == 27:  # ESC
            return False
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False  # window was closed with the X button


def build_head_move_targets():
    """On-screen dots to look at while slowly moving the head."""
    return [{"label": f"head_move_{x}_{y}", "x": x, "y": y, "on_screen": True,
             "head_move": True, "record_seconds": HEAD_MOVE_SECONDS}
            for x, y in HEAD_MOVE_POINTS]


def run_target(tracker, screen_size, target, rows):
    """Show one target for RECORD_SECONDS (or the target's own "record_seconds"),
    appending one row per frame to `rows`.
    Return False if the user pressed ESC so main() can stop.

    """
    w, h = screen_size
    settle = SETTLE_SECONDS if target["on_screen"] else OFF_SCREEN_SETTLE_SECONDS
    duration = target.get("record_seconds", RECORD_SECONDS)
    head_move = target.get("head_move", False)
    start = time.monotonic()

    while True:
        elapsed = time.monotonic() - start
        if elapsed >= settle + duration:
            break
        recording = elapsed >= settle

        # 0 during the settle phase, then 0 -> 1 over the recording time
        progress = min(max((elapsed - settle) / duration, 0.0), 1.0)

        frame, result = tracker.read()
        face_found = result is not None and bool(result.face_landmarks)

        if recording and frame is not None:
            row = {
                "label": target["label"],
                "x": target["x"],
                "y": target["y"],
                "on_screen": int(target["on_screen"]),
                "head_move": int(head_move),
                "t": round(elapsed - settle, 4),  # seconds into the recording
                "face_detected": int(face_found),
            }
            if face_found:
                row.update(extract_features(result))
            rows.append(row)

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if target["on_screen"]:
            # white ring = get ready, red shrinking ring = recording
            ring_color = (0, 0, 255) if recording else (255, 255, 255)
            draw_on_screen_target(canvas, target["x"] * w, target["y"] * h, progress, ring_color)
        else:
            draw_off_screen_target(canvas, target, progress)

        # only show text before recording, so it doesn't pull the eyes away mid-recording
        if head_move and not recording:
            text = "Keep your eyes on the dot and slowly move your head around"
            (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(canvas, text, (w // 2 - text_w // 2, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if not face_found and not recording:
            cv2.putText(canvas, "No face detected - check the camera", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

        cv2.imshow(WINDOW, canvas)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            return False

    if not target["on_screen"]:
        beep()  # tell the user to look back at the screen
    return True


def save(rows, info):
    """Write the recorded rows to a file in OUTPUT_DIR.

      - calibration_<date>_<time>.csv: one row per frame, one column per number.
        Frames with no face are kept with face_detected = 0 (feature columns blank).
      - calibration_<date>_<time>.json: `info`: screen size, camera resolution, settings.

    Returns the CSV path, or None if there was nothing to save.
    """
    if not rows:
        print("Nothing was recorded, so no file was saved.")
        return None

    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"calibration_{stamp}.csv"

    # column names = every key that appears in any row, in first-seen order
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    csv_path.with_suffix(".json").write_text(json.dumps(info, indent=2))

    with_face = sum(row["face_detected"] for row in rows)
    print(f"Saved {len(rows)} frames ({with_face} with a face detected) to {csv_path}")
    return csv_path


def main():
    screen_size = get_screen_size()
    # order: 9 on-screen dots -> 5 head-movement dots -> 8 off-screen targets
    basic = build_targets()
    targets = ([t for t in basic if t["on_screen"]] + build_head_move_targets()
               + [t for t in basic if not t["on_screen"]])
    tracker = FaceTracker()

    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "screen_size": list(screen_size),
        "camera_size": [int(tracker.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(tracker.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
        "camera_flipped": True,
        "record_seconds": RECORD_SECONDS,
        "settle_seconds": SETTLE_SECONDS,
        "off_screen_settle_seconds": OFF_SCREEN_SETTLE_SECONDS,
        "head_move_seconds": HEAD_MOVE_SECONDS,
        "targets_total": len(targets),
        "targets_completed": 0,
    }

    rows = []
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        started = show_message(screen_size, [
            "Gaze calibration",
            "",
            "Keep your head still and move only your eyes.",
            "Look at the red dot. When the ring turns red, recording starts -",
            "keep looking at the dot until the ring has shrunk.",
            "",
            "Press SPACE to start, ESC to quit at any time.",
        ])

        if started:
            head_intro_shown = False
            off_screen_intro_shown = False
            for target in targets:
                # pause once before the head-movement part to explain it
                if target.get("head_move") and not head_intro_shown:
                    head_intro_shown = True
                    if not show_message(screen_size, [
                        "Part 2: head movement",
                        "",
                        "Keep your eyes on the dot the whole time, but slowly move",
                        "your HEAD: turn left and right, tilt up and down, lean a bit.",
                        "Small, slow movements -- like you would while using the computer.",
                        "",
                        "Press SPACE to continue, ESC to quit.",
                    ]):
                        break

                # pause once before the off-screen part to explain it
                if not target["on_screen"] and not off_screen_intro_shown:
                    off_screen_intro_shown = True
                    if not show_message(screen_size, [
                        "Part 3: off-screen targets",
                        "",
                        "An arrow will point off the edge of the screen.",
                        "Look about 1 inch past the arrow tip, off the screen,",
                        "and keep looking there until you hear a beep.",
                        "",
                        "Press SPACE to continue, ESC to quit.",
                    ]):
                        break

                if not run_target(tracker, screen_size, target, rows):
                    print("Stopped early (ESC).")
                    break
                info["targets_completed"] += 1
    finally:
        # always runs so camera is always released
        tracker.close()
        cv2.destroyAllWindows()

    save(rows, info)


if __name__ == "__main__":
    main()
