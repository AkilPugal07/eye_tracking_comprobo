"""
Records how closed MediaPipe thinks each eye is while you wink,
blink and keep your eyes open. used to set threshold numbers.

saves calibration_data/wink_test_<date>_<time>.csv.
"""

import csv
import time
from datetime import datetime

import cv2
import numpy as np

import click_detector as cd
from calibration import OUTPUT_DIR, FaceTracker

GET_READY_SECONDS = 2.5
STEPS = [# (name, instruction, seconds)
    ("open", "Keep both eyes OPEN and look around the screen", 6),
    ("wink_left", "Wink your LEFT eye: close ~1 s, open, repeat", 8),
    ("wink_right", "Wink your RIGHT eye: close ~1 s, open, repeat", 8),
    ("blink", "Blink NORMALLY a few times", 6),
    ("closed", "Close BOTH eyes", 4),
]


def put(frame, text, y, color=(255, 255, 255), scale=0.8):
    font, margin = cv2.FONT_HERSHEY_SIMPLEX, 20
    max_w = frame.shape[1] - 2 * margin
    text_w = cv2.getTextSize(text, font, scale, 2)[0][0]
    if text_w > max_w:
        scale *= max_w / text_w
    cv2.putText(frame, text, (margin, y), font, scale, color, 2)


def record():
    tracker = FaceTracker()
    clicker = cd.ClickDetector()
    rows = []
    try:
        for step, instruction, seconds in STEPS:
            start = time.monotonic()
            while True:
                elapsed = time.monotonic() - start
                if elapsed >= GET_READY_SECONDS + seconds:
                    break
                recording = elapsed >= GET_READY_SECONDS
                frame, result = tracker.read()
                if frame is None:
                    raise SystemExit("No frame from camera.")
                button = clicker.update(result)

                if recording and clicker.left is not None:
                    rows.append({"step": step, "t": round(elapsed - GET_READY_SECONDS, 3),
                                 "left": clicker.left, "right": clicker.right,
                                 "would_click": button or ""})

                if recording:
                    put(frame, f"NOW ({GET_READY_SECONDS + seconds - elapsed:.0f} s left):", 35, (0, 200, 0))
                    put(frame, instruction, 70, (0, 200, 0))
                else:
                    put(frame, "NEXT:", 35, (0, 165, 255))
                    put(frame, instruction, 70, (0, 165, 255))
                if clicker.left is None:
                    put(frame, "no face detected", 110, (0, 0, 255))
                if button:
                    put(frame, f"{button.upper()} click would happen", 145, (0, 165, 255))
                cd.draw_blink_meter(frame, clicker.left, clicker.right)
                cv2.imshow("wink test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return rows
    finally:
        tracker.close()
        cv2.destroyAllWindows()
    return rows


def summarize(rows):
    print(f"\n{'step':12s} {'frames':>6s}   {'MediaPipe LEFT':>17s}   {'MediaPipe RIGHT':>17s}   {'left - right':>15s}   clicks")
    print(f"{'':12s} {'':>6s}   {'median / top 10%':>17s}   {'median / top 10%':>17s}   {'min / max':>15s}")
    for step, _, _ in STEPS:
        r = [x for x in rows if x["step"] == step]
        if not r:
            continue
        L = np.array([x["left"] for x in r]); R = np.array([x["right"] for x in r]); D = L - R
        clicks = [x["would_click"] for x in r if x["would_click"]]
        print(f"{step:12s} {len(r):6d}   {np.median(L):6.2f} / {np.percentile(L, 90):5.2f}     "
              f"{np.median(R):6.2f} / {np.percentile(R, 90):5.2f}     {D.min():+6.2f} / {D.max():+5.2f}   {clicks}")

    def clicked(step):
        return [x["would_click"] for x in rows if x["step"] == step and x["would_click"]]
    print()
    wl, wr = clicked("wink_left"), clicked("wink_right")
    false = clicked("open") + clicked("blink") + clicked("closed")
    if not wl and not wr:
        print("Winks were NOT detected with the current settings.")
    else:
        if wl and wr and set(wl) == {"right"} and set(wr) == {"left"}:
            print("winks detected, but LEFT/RIGHT are swapped: set SWAP_EYES = True in click_detector.py")
        else:
            print(f"left winks gave {wl or 'nothing'}, right winks gave {wr or 'nothing'}.")
    if false:
        print(f"{len(false)} click would have happened WITHOUT a wink bc thresholds too loose.")


def main():
    rows = record()
    if not rows:
        print("Nothing recorded.")
        return
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"wink_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows)
    print(f"\nSaved {len(rows)} frames to {path}")


if __name__ == "__main__":
    main()
