"""
Fit a gaze -> screen mapping from calibration data, and use it live.

Fit (uses the newest calibration CSV, or one you pass in):
    python gaze_boundary.py
    python gaze_boundary.py calibration_data/calibration_20261104_144608.csv
    python gaze_boundary.py --eyes-only        # old model without head pose (to compare)

Try it live with the webcam (needs a fitted model first):
    python gaze_boundary.py --live

How it works
------------
1. Features per frame:
     look_right, look_up   from MediaPipe's eye blendshapes (eyes relative to the face)
     head_yaw, head_pitch  head rotation in degrees, from the head pose matrix
   Head rotation is needed because when you turn your head while looking at the
   same spot, your eyes rotate the other way -- eyes alone would then say you're
   looking somewhere else.
2. Fit: least squares on every ON-screen frame (not just per-target averages,
   so head movements during calibration teach the head correction).
   The default model is curved (quadratic): besides
       screen_x = a*look_right + b*look_up + c*head_yaw + d*head_pitch + ...
   it also has squares and products (look_right*head_yaw, head_yaw**2, ...).
   A straight line only works for small head turns; with bigger ones the
   relationship bends. (--linear fits the straight-line model, to compare.)
   Tested alternatives that did NOT help: iris position relative to the nose,
   nose bridge, brows, forehead, chin, face sides, eye corners, whole face.
3. Bounding box: on screen  <=>  0 <= screen_x <= 1  and  0 <= screen_y <= 1
   (plus a small margin). The off-screen targets are only used to CHECK the box,
   because their labels (-0.2 / 1.2) are rough guesses.
4. Live: the position is smoothed with a One Euro filter -- heavy smoothing when
   the eyes are still (less jitter), light smoothing when they jump (less lag).
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

CALIBRATION_DIR = Path(__file__).with_name("calibration_data")
MODEL_PATH = CALIBRATION_DIR / "gaze_model.json"

BLINK_THRESHOLD = 0.5  # eyeBlink score above this = eyes closed, frame is ignored
DEFAULT_MARGIN = 0.03  # extra room around the screen (fraction of screen size)
SWITCH_FRAMES = 5      # frames in a row needed to switch on <-> off screen

# One Euro filter settings (tested on the calibration recording):
#   lower MIN_CUTOFF = less jitter when still, but more lag
#   higher BETA      = follows fast eye jumps quicker, but a bit more jitter
FILTER_MIN_CUTOFF = 0.5
FILTER_BETA = 0.5

EYE_FEATURES = ["look_right", "look_up"]
HEAD_FEATURES = ["head_yaw", "head_pitch"]

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


def head_angles(fx, fy, fz):
    """Head yaw and pitch in degrees, from the direction the face points.

    (fx, fy, fz) is the third column of the 4x4 head pose matrix
    (m02, m12, m22): the face's forward axis in camera coordinates.
    """
    yaw = math.degrees(math.atan2(fx, fz))
    pitch = math.degrees(math.atan2(fy, math.hypot(fx, fz)))
    return yaw, pitch


def features_from_result(result):
    """Live MediaPipe result -> dict of all features, or None if no face / blinking."""
    if result is None or not result.face_blendshapes:
        return None
    scores = {shape.category_name: shape.score for shape in result.face_blendshapes[0]}
    if is_blinking(scores):
        return None
    feats = dict(zip(EYE_FEATURES, gaze_features(scores)))
    if result.facial_transformation_matrixes:
        m = np.asarray(result.facial_transformation_matrixes[0])
        feats.update(zip(HEAD_FEATURES, head_angles(m[0, 2], m[1, 2], m[2, 2])))
    return feats


# ---------------------------------------------------------------- smoothing

class OneEuroFilter:
    """Adaptive low-pass filter for one number (Casiez et al., 2012).

    Smooths a lot when the value barely changes (removes jitter), and less
    when it changes fast (so quick eye jumps don't lag behind).
    """

    def __init__(self, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._t = None
        self._x = None
        self._dx = 0.0

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._t is None:
            self._t, self._x = t, x
            return x
        dt = max(t - self._t, 1e-3)
        self._t = t
        # how fast is the value changing? (itself smoothed)
        a_d = self._alpha(dt, self.d_cutoff)
        self._dx = a_d * (x - self._x) / dt + (1 - a_d) * self._dx
        # faster change -> higher cutoff -> less smoothing
        a = self._alpha(dt, self.min_cutoff + self.beta * abs(self._dx))
        self._x = a * x + (1 - a) * self._x
        return self._x


# ---------------------------------------------------------------- fitting

def newest_calibration_csv():
    files = sorted(CALIBRATION_DIR.glob("calibration_*.csv"))  # names sort by date
    if not files:
        raise SystemExit(f"No calibration CSV found in {CALIBRATION_DIR}. Run calibration.py first.")
    return files[-1]


def load_frames(csv_path):
    """Read the calibration CSV -> list of frames with all features.

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
            frame = {
                "label": row["label"],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "on_screen": row["on_screen"] == "1",
                "head_move": row.get("head_move") == "1",
            }
            frame.update(zip(EYE_FEATURES, gaze_features(scores)))
            frame.update(zip(HEAD_FEATURES, head_angles(
                float(row["head_m02"]), float(row["head_m12"]), float(row["head_m22"]))))
            frames.append(frame)
    n_head = sum(f["head_move"] for f in frames)
    print(f"Loaded {len(frames)} frames (skipped {no_face} with no face, {blinks} blinks)"
          + (f", {n_head} from the head-movement part" if n_head else ""))
    if not n_head:
        print("  (No head-movement part in this file -- the head correction is only roughly")
        print("   learned. Re-run calibration.py to record one.)")
    return frames


def design_row(feats, feature_names, center=None, degree=1):
    """The model's input terms for one frame.

    degree 1 (straight-line model): the features, then a 1 for the constant.
    degree 2 (curved model): also every square and pair product, e.g.
        look_right**2, look_right*head_yaw, ... -- this lets the fit bend, which
        matters once the head turns more than a few degrees.
    Features are centered first (minus their typical value) so the squares
    measure "how far from normal" rather than raw size.
    """
    center = center or {}
    base = [feats[n] - center.get(n, 0.0) for n in feature_names]
    terms = list(base)
    if degree >= 2:
        for i in range(len(base)):
            for j in range(i, len(base)):
                terms.append(base[i] * base[j])
    return terms + [1.0]


def fit_mapping(frames, feature_names, degree=2):
    """Least-squares fit of screen_x and screen_y from every ON-screen frame.

    Returns (coef_x, coef_y, center): one coefficient per model term (see
    design_row), and the typical value of each feature (the median while
    looking at the normal dots with the head still).
    """
    on = [f for f in frames if f["on_screen"]]
    if len({f["label"] for f in on}) < 3:
        raise SystemExit("Need at least 3 on-screen targets to fit.")
    still = [f for f in on if not f["head_move"]] or on
    center = {n: float(np.median([f[n] for f in still])) for n in feature_names}
    A = np.array([design_row(f, feature_names, center, degree) for f in on])
    coef_x, *_ = np.linalg.lstsq(A, np.array([f["x"] for f in on]), rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, np.array([f["y"] for f in on]), rcond=None)
    return coef_x, coef_y, center


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
    (calibration.py and main.py both flip it), and for the head features the
    detector needs output_facial_transformation_matrixes=True
    (calibration.FaceTracker has that).
    """

    def __init__(self, coef_x, coef_y, features=EYE_FEATURES, center=None, degree=1,
                 margin=DEFAULT_MARGIN, switch_frames=SWITCH_FRAMES,
                 min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA):
        self.coef_x = np.asarray(coef_x, dtype=float)
        self.coef_y = np.asarray(coef_y, dtype=float)
        self.features = list(features)
        self.center = dict(center or {})
        self.degree = degree
        self.margin = margin
        self.switch_frames = switch_frames
        self._filter_x = OneEuroFilter(min_cutoff, beta)
        self._filter_y = OneEuroFilter(min_cutoff, beta)

        self.on_screen = True  # current (debounced) state
        self._disagree = 0     # frames in a row that disagree with the state

    @classmethod
    def load(cls, path=MODEL_PATH, **kwargs):
        model = json.loads(Path(path).read_text())
        kwargs.setdefault("margin", model.get("margin", DEFAULT_MARGIN))
        # older models (before head pose / the curved model) default to the old behavior
        kwargs.setdefault("features", model.get("features_used", EYE_FEATURES))
        kwargs.setdefault("center", model.get("center", {}))
        kwargs.setdefault("degree", model.get("degree", 1))
        return cls(model["coef_x"], model["coef_y"], **kwargs)

    def to_screen(self, feats):
        """Feature dict -> screen position as fractions (0..1 = on screen)."""
        v = np.array(design_row(feats, self.features, self.center, self.degree))
        return float(self.coef_x @ v), float(self.coef_y @ v)

    def is_inside(self, x, y):
        m = self.margin
        return -m <= x <= 1 + m and -m <= y <= 1 + m

    def update(self, result, t=None):
        """Feed one MediaPipe result. Returns None if no face or blinking,
        otherwise {"x", "y", "on_screen", "inside_now"}.

        x, y       smoothed screen position (fractions; can be <0 or >1)
        inside_now raw check for this frame only
        on_screen  debounced: only flips after SWITCH_FRAMES frames in a row,
                   so it doesn't flicker at the edges
        """
        feats = features_from_result(result)
        if feats is None or any(n not in feats for n in self.features):
            return None

        t = time.monotonic() if t is None else t
        raw_x, raw_y = self.to_screen(feats)
        x, y = self._filter_x(raw_x, t), self._filter_y(raw_y, t)

        inside_now = self.is_inside(x, y)
        if inside_now != self.on_screen:
            self._disagree += 1
            if self._disagree >= self.switch_frames:
                self.on_screen = inside_now
                self._disagree = 0
        else:
            self._disagree = 0

        return {"x": x, "y": y, "on_screen": self.on_screen, "inside_now": inside_now}


# ---------------------------------------------------------------- report

def evaluate(frames, mapper):
    """Print how well the box separates on- and off-screen, and return a summary."""
    print("\nPer target (median predicted position, % of frames classified correctly):")
    print(f"  {'label':22s} {'true (x, y)':>13s}   {'predicted':>13s}   correct")
    by_label = {}
    for fr in frames:
        by_label.setdefault(fr["label"], []).append(fr)

    summary = []
    weak = []
    for label, frs in by_label.items():
        t = frs[0]
        preds = [mapper.to_screen(f) for f in frs]
        correct = float(np.mean([mapper.is_inside(px, py) == t["on_screen"] for px, py in preds]))
        px = float(np.median([p[0] for p in preds]))
        py = float(np.median([p[1] for p in preds]))
        flag = "" if correct >= 0.9 else "   <-- weak"
        print(f"  {label:22s} ({t['x']:4.2g}, {t['y']:4.2g})   ({px:5.2f}, {py:5.2f})   {correct:6.1%}{flag}")
        summary.append({"label": label, "x": t["x"], "y": t["y"], "on_screen": t["on_screen"],
                        "pred_x": px, "pred_y": py, "correct": correct, "frames": len(frs)})
        if correct < 0.9:
            weak.append(t)

    inside = [mapper.is_inside(*mapper.to_screen(f)) for f in frames]
    on_acc = float(np.mean([i for i, f in zip(inside, frames) if f["on_screen"]]))
    off = [not i for i, f in zip(inside, frames) if not f["on_screen"]]
    off_acc = float(np.mean(off)) if off else float("nan")

    def typical_error(group):  # median distance from the true target, as % of the screen
        errs = [math.hypot(px - f["x"], py - f["y"])
                for f in group for px, py in [mapper.to_screen(f)]]
        return float(np.median(errs)) if errs else float("nan")
    still_err = typical_error([f for f in frames if f["on_screen"] and not f["head_move"]])
    move_err = typical_error([f for f in frames if f["head_move"]])

    print(f"\nTypical error, head still:  {still_err:.1%} of the screen")
    if not math.isnan(move_err):
        print(f"Typical error, head moving: {move_err:.1%} of the screen")
    print(f"On-screen frames classified ON:   {on_acc:.1%}")
    print(f"Off-screen frames classified OFF: {off_acc:.1%}")

    if weak:
        print("\nWeak spots (under 90% correct):")
        for t in weak:
            print(f"  - {t['label']} ({describe_position(t['x'], t['y'])})")
        if any(t["y"] < 0 for t in weak):
            print("  Looking ABOVE the screen looks too similar to the top row of dots.")
            print("  Try re-calibrating with a sticky note well above the screen to look at,")
            print("  or use a bigger margin.")
    return summary, on_acc, off_acc


def fit_and_save(csv_path, margin, use_head=True, degree=2):
    print(f"Calibration file: {csv_path}")
    frames = load_frames(csv_path)
    feature_names = EYE_FEATURES + (HEAD_FEATURES if use_head else [])
    coef_x, coef_y, center = fit_mapping(frames, feature_names, degree)
    mapper = GazeMapper(coef_x, coef_y, features=feature_names, center=center,
                        degree=degree, margin=margin)

    # fit quality: error of each on-screen target's median prediction
    errs = []
    for label in dict.fromkeys(f["label"] for f in frames if f["on_screen"]):
        frs = [f for f in frames if f["label"] == label]
        preds = np.array([mapper.to_screen(f) for f in frs])
        errs.append(max(abs(np.median(preds[:, 0]) - frs[0]["x"]), abs(np.median(preds[:, 1]) - frs[0]["y"])))
    err = float(max(errs))

    shape = "curved (quadratic)" if degree >= 2 else "straight-line"
    print(f"\n{shape} fit using {', '.join(feature_names)} ({len(coef_x)} terms)")
    print(f"  worst on-screen target error {err:.3f} of the screen")
    if degree == 1:
        for axis, coef in (("x", coef_x), ("y", coef_y)):
            terms = " ".join(f"{c:+.3f}*{n}" for c, n in zip(coef, feature_names))
            print(f"  screen_{axis} = {terms} {coef[-1]:+.3f}")
    print(f"  on screen if both are between {-margin:.2f} and {1 + margin:.2f}")

    summary, on_acc, off_acc = evaluate(frames, mapper)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_csv": Path(csv_path).name,
        "features_used": feature_names,
        "degree": degree,
        "center": center,
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
    parser.add_argument("--eyes-only", action="store_true",
                        help="fit without head pose, to compare")
    parser.add_argument("--linear", action="store_true",
                        help="straight-line model instead of the curved one, to compare")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                        help=f"extra room around the screen, fraction (default {DEFAULT_MARGIN})")
    args = parser.parse_args()

    if args.live:
        run_live(args.margin)
    else:
        fit_and_save(args.csv or newest_calibration_csv(), args.margin,
                     use_head=not args.eyes_only, degree=1 if args.linear else 2)


if __name__ == "__main__":
    main()
