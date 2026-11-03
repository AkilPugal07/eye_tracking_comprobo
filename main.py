import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Download a face landmarker model bundle if you haven't already:
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# Place 'face_landmarker.task' in the same directory as this script.

base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1,
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)


while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    flipped_img = cv2.flip(
        img, 1
    )  # Flip the image horizontally for a selfie-view display

    # MediaPipe Tasks expects an mp.Image object
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(flipped_img, cv2.COLOR_BGR2RGB),
    )

    # Run face landmark detection
    detection_result = detector.detect(mp_image)

    # Draw landmarks manually if any faces are detected
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            # Draw each landmark point
            for landmark in face_landmarks:

                # left_eye = face_landmarks[468]  # Example: Left eye iris landmark index
                right_eye = face_landmarks[
                    473
                ]  # Example: Right eye iris landmark index

                left_eye_x = int(left_eye.x * img.shape[1])
                left_eye_y = int(left_eye.y * img.shape[0])

                right_eye_x = int(right_eye.x * img.shape[1])
                right_eye_y = int(right_eye.y * img.shape[0])

                cv2.circle(flipped_img, (left_eye_x, left_eye_y), 5, (0, 255, 0), -1)
                cv2.circle(flipped_img, (right_eye_x, right_eye_y), 5, (0, 255, 0), -1)

    cv2.imshow("Face Mesh", flipped_img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
