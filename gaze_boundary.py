"""
Fits a model from the calibration data that maps where your eyes are looking
to a position on the screen. also decides if you're looking on or off the screen.

    python gaze_boundary.py #fit on the newest calibration
    python gaze_boundary.py calibration_data/calibration_<date>_<time>.csv #fit on a specific one
    python gaze_boundary.py --live #test it with the webcam
    --eyes-only and --linear fit the older simpler models to compare

features used for each frame:
    look_right, look_up: from mediapipe's eye blendshapes
    head_yaw, head_pitch: head rotation in degrees from the head pose matrix
head rotation is needed bc if you turn your head while looking at the same spot
your eyes rotate the other way, so eyes alone would think you looked somewhere else.

the model is quadratic bc a straight line only worked for small head turns. 
it's fit on all the on screen frames. on screen means the predicted x and y are both between 0 and 1 with a small margin.
off screen targets are only used to check it since their positions are rough guesses.
"""

import argparse
import contextlib
import csv
import io
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

CALIBRATION_DIR = Path(__file__).with_name("calibration_data")
MODEL_PATH = CALIBRATION_DIR / "gaze_model.json"

BLINK_THRESHOLD = 0.5 #frames with eyeBlink above this get ignored
DEFAULT_MARGIN = 0.03 #extra room around the screen
SWITCH_FRAMES = 5 #frames in a row needed to switch between on and off screen

#One Euro filter settings. lower min cutoff = less jitter but more lag,
#higher beta = follows quick eye movements faster but more jitter
FILTER_MIN_CUTOFF = 0.5
FILTER_BETA = 0.5

EYE_FEATURES = ["look_right", "look_up"]
HEAD_FEATURES = ["head_yaw", "head_pitch"]

BLENDSHAPES_USED = [
    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeBlinkLeft", "eyeBlinkRight",
]


#features

def is_blinking(scores):
    return max(scores["eyeBlinkLeft"], scores["eyeBlinkRight"]) > BLINK_THRESHOLD


def gaze_features(scores):
    """turns the blendshape scores into (look_right, look_up).

    looking right means the left eye looks in and the right eye looks out.
    the sign doesn't really matter since the fit figures it out.
    """
    look_right = (scores["eyeLookInLeft"] + scores["eyeLookOutRight"]) \
               - (scores["eyeLookOutLeft"] + scores["eyeLookInRight"])
    look_up = (scores["eyeLookUpLeft"] + scores["eyeLookUpRight"]) \
            - (scores["eyeLookDownLeft"] + scores["eyeLookDownRight"])
    return look_right, look_up


def head_angles(fx, fy, fz):
    """head yaw and pitch in degrees. (fx, fy, fz) is the third column of the head pose
    matrix (m02, m12, m22), which is the direction the face is pointing"""
    yaw = math.degrees(math.atan2(fx, fz))
    pitch = math.degrees(math.atan2(fy, math.hypot(fx, fz)))
    return yaw, pitch


def features_from_result(result):
    """gets the features from a live mediapipe result. None if no face or blinking"""
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


#smoothing

class OneEuroFilter:
    """One Euro filter (Casiez et al. 2012). smooths a lot when the value is
    barely changing and less when it changes fast, so there's less jitter
    without the cursor lagging behind quick eye movements."""

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
        #how fast the value is changing (also smoothed)
        a_d = self._alpha(dt, self.d_cutoff)
        self._dx = a_d * (x - self._x) / dt + (1 - a_d) * self._dx
        #faster change means less smoothing
        a = self._alpha(dt, self.min_cutoff + self.beta * abs(self._dx))
        self._x = a * x + (1 - a) * self._x
        return self._x


#fitting

def is_complete_calibration(csv_path):
    """False if the calibration was stopped early with ESC"""
    info_path = Path(csv_path).with_suffix(".json")
    try:
        info = json.loads(info_path.read_text())
        return info.get("targets_completed", 0) >= info.get("targets_total", 0)
    except (OSError, json.JSONDecodeError):
        return True  #no json file, assume it's fine


def newest_calibration_csv():
    """newest complete calibration. file names have the date and time so sorting works"""
    files = [f for f in sorted(CALIBRATION_DIR.glob("calibration_*.csv")) if is_complete_calibration(f)]
    if not files:
        raise SystemExit(f"No complete calibration found in {CALIBRATION_DIR}. Run calibration.py first.")
    return files[-1]


def ensure_model_is_current(margin=DEFAULT_MARGIN):
    """refits the model if there's a newer calibration than the one it used, or if
    there's no model or it's broken. main.py calls this when it starts."""
    newest = newest_calibration_csv()
    source = None
    if MODEL_PATH.exists():
        try:
            source = json.loads(MODEL_PATH.read_text()).get("source_csv")
        except (OSError, json.JSONDecodeError):
            pass  #broken file (like from a merge conflict) so refit
    if source == newest.name:
        return
    print(f"Fitting the gaze model on the newest calibration ({newest.name})...")
    report = io.StringIO()
    with contextlib.redirect_stdout(report):  #full report is long so only print the main numbers
        fit_and_save(newest, margin)
    for line in report.getvalue().splitlines():
        if line.startswith(("Typical error", "On-screen frames", "Off-screen frames")):
            print("  " + line)


def load_frames(csv_path):
    """reads a calibration csv into a list of frames with features.
    skips frames with no face or a blink"""
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
    """the terms that go into the model for one frame.

    degree 1 is just the features plus a 1 for the constant.
    degree 2 also adds every square and product of two features.
    features get centered first so the squares mean how far from normal you are.
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
    """least squares fit of screen x and y using all the on screen frames.
    returns coef_x, coef_y, and center (the median of each feature with the head still)"""
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
    #ex. (-0.2, -0.2) gives "above-left"
    vert = "above" if y < 0 else "below" if y > 1 else ""
    horiz = "left" if x < 0 else "right" if x > 1 else ""
    return "-".join(p for p in (vert, horiz) if p) or "on screen"


#live

class GazeMapper:
    """Takes live mediapipe results and gives a screen position and whether
    you're looking on screen. main.py uses this.

    the camera image has to be flipped the same way as in calibration, and the
    detector needs output_facial_transformation_matrixes=True for the head features.
    calibration.FaceTracker does both.
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

        self.on_screen = True
        self._disagree = 0 #frames in a row that disagree with on_screen

    @classmethod
    def load(cls, path=MODEL_PATH, **kwargs):
        model = json.loads(Path(path).read_text())
        kwargs.setdefault("margin", model.get("margin", DEFAULT_MARGIN))
        #older model files don't have these so use the old defaults
        kwargs.setdefault("features", model.get("features_used", EYE_FEATURES))
        kwargs.setdefault("center", model.get("center", {}))
        kwargs.setdefault("degree", model.get("degree", 1))
        return cls(model["coef_x"], model["coef_y"], **kwargs)

    def to_screen(self, feats):
        #screen position as fractions, 0 to 1 is on screen
        v = np.array(design_row(feats, self.features, self.center, self.degree))
        return float(self.coef_x @ v), float(self.coef_y @ v)

    def is_inside(self, x, y):
        m = self.margin
        return -m <= x <= 1 + m and -m <= y <= 1 + m

    def update(self, result, t=None):
        """call once per frame. returns None if no face or blinking, otherwise a dict with
        x, y: smoothed screen position (can go below 0 or above 1)
        inside_now: if this frame alone is on screen
        on_screen: only changes after SWITCH_FRAMES frames in a row so it doesnt flicker at the edges
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


#report

def evaluate(frames, mapper):
    """prints how accurate the model is for each target and overall"""
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

    def typical_error(group):  #median distance from the actual target
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

    #worst error out of all the on screen targets
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
    """webcam test that shows ON/OFF SCREEN and a small map of where you're looking"""
    import cv2
    from calibration import FaceTracker

    ensure_model_is_current(margin)
    mapper = GazeMapper.load(margin=margin)
    tracker = FaceTracker()

    #map goes from -0.5 to 1.5, the box in the middle is the screen
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
