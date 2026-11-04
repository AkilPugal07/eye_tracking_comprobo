import cv2
import numpy as np
import calibration as c

# ---- Test 1: screen size + every target, fullscreen ----
# Press any key for the next target, ESC to skip ahead to the camera test.
w, h = c.get_screen_size()
print("screen size:", w, h)

cv2.namedWindow("test", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

for target in c.build_targets():
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if target["on_screen"]:
        c.draw_on_screen_target(canvas, target["x"] * w, target["y"] * h, 0.5)
    else:
        c.draw_off_screen_target(canvas, target, 0.5)
    cv2.imshow("test", canvas)
    if cv2.waitKey(0) & 0xFF == 27:  # 27 = ESC
        break
cv2.destroyAllWindows()

# ---- Test 2: camera + face detection ----
# Press q to quit.
tracker = c.FaceTracker()
try:
    while True:
        frame, result = tracker.read()
        if frame is None:
            print("no frame from camera")
            break
        found = bool(result.face_landmarks)
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(frame, f"face found: {found}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("camera test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    tracker.close()
    cv2.destroyAllWindows()
