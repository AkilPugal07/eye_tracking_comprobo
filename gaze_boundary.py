"""
Fit a gaze -> screen mapping from calibration data, and use it live.

Fit (uses the newest calibration CSV, or one you pass in):
    python gaze_boundary.py
    python gaze_boundary.py calibration_data/calibration_20261104_144608.csv

Try it live with the webcam (needs a fitted model first):
    python gaze_boundary.py --live

How it works
------------
1. Gaze features: two numbers per frame from MediaPipe's eye blendshapes,
   look_right and look_up. Blendshapes measure the eyes relative to the face,
   so small head movements don't shift them (raw iris positions in the camera
   image do shift, which is why the iris midpoint doesn't work).
2. Per target: drop blink frames and take the median -> one point per target.
3. Fit: least squares on the 9 ON-screen targets only (their labels are exact):
       screen_x = a*look_right + b*look_up + c
       screen_y = d*look_right + e*look_up + f
4. Bounding box: on screen  <=>  0 <= screen_x <= 1  and  0 <= screen_y <= 1
   (plus a small margin). The off-screen targets are only used to CHECK the box,
   because their labels (-0.2 / 1.2) are rough guesses.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

CALIBRATION_DIR = Path(__file__).with_name("calibration_data")
MODEL_PATH = CALIBRATION_DIR / "gaze_model.json"

BLINK_THRESHOLD = 0.5  # eyeBlink score above this = eyes closed, frame is ignored
DEFAULT_MARGIN = 0.03  # extra room around the screen (fraction of screen size)
SWITCH_FRAMES = 5      # frames in a row needed to switch on <-> off screen

# the blendshapes the gaze features are built from
BLENDSHAPES_USED = [
    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeBlinkLeft", "eyeBlinkRight",
]


# ---------------------------------------------------------------- features

def is_blinking(scores):
    """scores: dict of blendshape name -> score (0..1)."""
    return max(scores["eyeBlinkLeft"], scores["eyeBlinkRight"]) > BLINK_THRESHOLD


def gaze_features(scores):
    """Turn blendshape scores into two gaze numbers: (look_right, look_up).

    Both eyes move the same way, so e.g. "looking right" = left eye looks
    in (toward the nose) + right eye looks out. Exact sign/direction doesn't
    matter -- the fit learns it (your camera is mirrored, for example).
    """
    look_right = (scores["eyeLookInLeft"] + scores["eyeLookOutRight"]) \
               - (scores["eyeLookOutLeft"] + scores["eyeLookInRight"])
    look_up = (scores["eyeLookUpLeft"] + scores["eyeLookUpRight"]) \
            - (scores["eyeLookDownLeft"] + scores["eyeLookDownRight"])
    return look_right, look_up


def scores_from_result(result):
    """Blendshape scores from a live MediaPipe result, or None if no face."""
    if result is None or not result.face_blendshapes:
        return None
    return {shape.category_name: shape.score for shape in result.face_blendshapes[0]}


# ---------------------------------------------------------------- fitting

def newest_calibration_csv():
    files = sorted(CALIBRATION_DIR.glob("calibration_*.csv"))  # names sort by date
    if not files:
        raise SystemExit(f"No calibration CSV found in {CALIBRATION_DIR}. Run calibration.py first.")
    return files[-1]


def load_frames(csv_path):
    """Read the calibration CSV -> list of frames with gaze features.

    Skips frames with no face and frames where the user blinked.
    """
    frames = []
    no_face = blinks = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["face_detected"] != "1":
                no_face += 1
                continue
            scores = {name: float(row["bs_" + name]) for name in BLENDSHAPES_USED}
            if is_blinking(scores):
                blinks += 1
                continue
            look_right, look_up = gaze_features(scores)
            frames.append({
                "label": row["label"],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "on_screen": row["on_screen"] == "1",
                "look_right": look_right,
                "look_up": look_up,
            })
    print(f"Loaded {len(frames)} frames (skipped {no_face} with no face, {blinks} blinks)")
    return frames


def per_target_medians(frames):
    """One point per target: the median gaze features of its frames."""
    targets = {}
    for fr in frames:
        targets.setdefault(fr["label"], []).append(fr)
    out = []
    for label, frs in targets.items():  # dicts keep insertion order = recording order
        out.append({
            "label": label,
            "x": frs[0]["x"],
            "y": frs[0]["y"],
            "on_screen": frs[0]["on_screen"],
            "look_right": float(np.median([f["look_right"] for f in frs])),
            "look_up": float(np.median([f["look_up"] for f in frs])),
            "frames": len(frs),
        })
    return out


def fit_mapping(targets):
    """Least-squares fit of screen_x and screen_y from the ON-screen targets.

    Returns (coef_x, coef_y), each [a, b, c] for  a*look_right + b*look_up + c.
    """
    on = [t for t in targets if t["on_screen"]]
    if len(on) < 3:
        raise SystemExit("Need at least 3 on-screen targets to fit.")
    A = np.array([[t["look_right"], t["look_up"], 1.0] for t in on])
    coef_x, *_ = np.linalg.lstsq(A, np.array([t["x"] for t in on]), rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, np.array([t["y"] for t in on]), rcond=None)
    return coef_x, coef_y


def describe_position(x, y):
    """(-0.2, -0.2) -> 'above-left' etc., for readable reports."""
    vert = "above" if y < 0 else "below" if y > 1 else ""
    horiz = "left" if x < 0 else "right" if x > 1 else ""
    return "-".join(p for p in (vert, horiz) if p) or "on screen"


# ---------------------------------------------------------------- live use

class GazeMapper:
    """Turns live MediaPipe results into a screen position + on/off screen.

    Usage:
        mapper = GazeMapper.load()
        out = mapper.update(result)     # result from detector.detect(...)
        if out is not None and out["on_screen"]:
            px, py = out["x"] * screen_w, out["y"] * screen_h

    The image must be flipped the same way as during calibration
    (calibration.py and main.py both flip it).
    """

    def __init__(self, coef_x, coef_y, margin=DEFAULT_MARGIN,
                 switch_frames=SWITCH_FRAMES, smoothing=0.5):
        self.coef_x = np.asarray(coef_x, dtype=float)
        self.coef_y = np.asarray(coef_y, dtype=float)
        self.margin = margin
        self.switch_frames = switch_frames
        self.smoothing = smoothing  # 0 = raw, closer to 1 = smoother but laggier

        self.on_screen = True  # current (debounced) state
        self._disagree = 0     # frames in a row that disagree with the state
        self._xy = None        # smoothed position

    @classmethod
    def load(cls, path=MODEL_PATH, **kwargs):
        model = json.loads(Path(path).read_text())
        kwargs.setdefault("margin", model.get("margin", DEFAULT_MARGIN))
        return cls(model["coef_x"], model["coef_y"], **kwargs)

    def to_screen(self, look_right, look_up):
        """Gaze features -> screen position as fractions (0..1 = on screen)."""
        v = np.array([look_right, look_up, 1.0])
        return float(self.coef_x @ v), float(self.coef_y @ v)

    def is_inside(self, x, y):
        m = self.margin
        return -m <= x <= 1 + m and -m <= y <= 1 + m

    def update(self, result):
        """Feed one MediaPipe result. Returns None if no face or blinking,
        otherwise {"x", "y", "on_screen", "inside_now"}.

        x, y       smoothed screen position (fractions; can be <0 or >1)
        inside_now raw check for this frame only
        on_screen  debounced: only flips after SWITCH_FRAMES frames in a row,
                   so it doesn't flicker at the edges
        """
        scores = scores_from_result(result)
        if scores is None or is_blinking(scores):
            return None

        x, y = self.to_screen(*gaze_features(scores))
        if self._xy is None:
            self._xy = (x, y)
        else:
            s = self.smoothing
            self._xy = (s * self._xy[0] + (1 - s) * x, s * self._xy[1] + (1 - s) * y)

        inside_now = self.is_inside(*self._xy)
        if inside_now != self.on_screen:
            self._disagree += 1
            if self._disagree >= self.switch_frames:
                self.on_screen = inside_now
                self._disagree = 0
        else:
            self._disagree = 0

        return {"x": self._xy[0], "y": self._xy[1],
                "on_screen": self.on_screen, "inside_now": inside_now}


# ---------------------------------------------------------------- report

def evaluate(frames, targets, mapper):
    """Print how well the box separates on- and off-screen, and return a summary."""
    print("\nPer target (median predicted position, % of frames classified correctly):")
    print(f"  {'label':22s} {'true (x, y)':>13s}   {'predicted':>13s}   correct")
    by_label = {}
    for fr in frames:
        by_label.setdefault(fr["label"], []).append(fr)

    summary = []
    weak = []
    for t in targets:
        preds = [mapper.to_screen(f["look_right"], f["look_up"]) for f in by_label[t["label"]]]
        correct = np.mean([mapper.is_inside(px, py) == t["on_screen"] for px, py in preds])
        px = float(np.median([p[0] for p in preds]))
        py = float(np.median([p[1] for p in preds]))
        flag = "" if correct >= 0.9 else "   <-- weak"
        print(f"  {t['label']:22s} ({t['x']:4.1f}, {t['y']:4.1f})   ({px:5.2f}, {py:5.2f})   {correct:6.1%}{flag}")
        summary.append({**t, "pred_x": px, "pred_y": py, "correct": float(correct)})
        if correct < 0.9:
            weak.append(t)

    on_frames = [f for f in frames if f["on_screen"]]
    off_frames = [f for f in frames if not f["on_screen"]]
    on_acc = np.mean([mapper.is_inside(*mapper.to_screen(f["look_right"], f["look_up"])) for f in on_frames])
    off_acc = np.mean([not mapper.is_inside(*mapper.to_screen(f["look_right"], f["look_up"])) for f in off_frames]) \
        if off_frames else float("nan")
    print(f"\nOn-screen frames classified ON:   {on_acc:.1%}")
    print(f"Off-screen frames classified OFF: {off_acc:.1%}")

    if weak:
        print("\nWeak spots (under 90% correct):")
        for t in weak:
            where = describe_position(t["x"], t["y"])
            print(f"  - {t['label']} ({where})")
        if any(t["y"] < 0 for t in weak):
            print("  Looking ABOVE the screen looks too similar to the top row of dots.")
            print("  Try re-calibrating with a sticky note well above the screen to look at,")
            print("  or use a bigger margin for the top edge.")
    return summary, float(on_acc), float(off_acc)


def fit_and_save(csv_path, margin):
    print(f"Calibration file: {csv_path}")
    frames = load_frames(csv_path)
    targets = per_target_medians(frames)
    coef_x, coef_y = fit_mapping(targets)

    mapper = GazeMapper(coef_x, coef_y, margin=margin)
    on = [t for t in targets if t["on_screen"]]
    err = max(max(abs(mapper.to_screen(t["look_right"], t["look_up"])[0] - t["x"]),
                  abs(mapper.to_screen(t["look_right"], t["look_up"])[1] - t["y"])) for t in on)
    print(f"\nFit on {len(on)} on-screen targets: worst error {err:.3f} of the screen size")
    print(f"  screen_x = {coef_x[0]:+.3f}*look_right {coef_x[1]:+.3f}*look_up {coef_x[2]:+.3f}")
    print(f"  screen_y = {coef_y[0]:+.3f}*look_right {coef_y[1]:+.3f}*look_up {coef_y[2]:+.3f}")
    print(f"  on screen if both are between {-margin:.2f} and {1 + margin:.2f}")

    summary, on_acc, off_acc = evaluate(frames, targets, mapper)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_csv": Path(csv_path).name,
        "features": "look_right, look_up from eye blendshapes (see gaze_features)",
        "coef_x": coef_x.tolist(),
        "coef_y": coef_y.tolist(),
        "margin": margin,
        "fit_worst_error": err,
        "on_screen_accuracy": on_acc,
        "off_screen_accuracy": off_acc,
        "targets": summary,
    }, indent=2))
    print(f"\nSaved model to {MODEL_PATH}")


def run_live(margin):
    """Webcam test: shows ON/OFF SCREEN and a mini map of where you're looking."""
    import cv2
    from calibration import FaceTracker  # same camera + detector setup, same flip

    if not MODEL_PATH.exists():
        raise SystemExit("No gaze model yet -- run `python gaze_boundary.py` first.")
    mapper = GazeMapper.load(margin=margin)
    tracker = FaceTracker()

    # mini map: shows x and y from -0.5 to 1.5, the screen is the box in the middle
    map_w, map_h = 200, 150
    def to_map(x, y):
        mx = int((min(max(x, -0.5), 1.5) + 0.5) / 2 * map_w)
        my = int((min(max(y, -0.5), 1.5) + 0.5) / 2 * map_h)
        return mx, my

    print("Press q to quit.")
    try:
        while True:
            frame, result = tracker.read()
            if frame is None:
                print("No frame from camera")
                break
            out = mapper.update(result)

            mini = np.zeros((map_h, map_w, 3), dtype=np.uint8)
            cv2.rectangle(mini, to_map(0, 0), to_map(1, 1), (255, 255, 255), 1)
            if out is None:
                text, color = "no face / blinking", (0, 165, 255)
            else:
                on = out["on_screen"]
                text = f"ON SCREEN ({out['x']:.2f}, {out['y']:.2f})" if on else "OFF SCREEN"
                color = (0, 200, 0) if on else (0, 0, 255)
                cv2.circle(mini, to_map(out["x"], out["y"]), 5, color, -1)

            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            h, w = frame.shape[:2]
            frame[10:10 + map_h, w - map_w - 10:w - 10] = mini
            cv2.imshow("gaze test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Fit / test the gaze -> screen boundary.")
    parser.add_argument("csv", nargs="?", help="calibration CSV (default: newest one)")
    parser.add_argument("--live", action="store_true", help="test the saved model with the webcam")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                        help=f"extra room around the screen, fraction (default {DEFAULT_MARGIN})")
    args = parser.parse_args()

    if args.live:
        run_live(args.margin)
    else:
        fit_and_save(args.csv or newest_calibration_csv(), args.margin)


if __name__ == "__main__":
    main()
