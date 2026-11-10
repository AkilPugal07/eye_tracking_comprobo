# Eye-tracking mouse

**Akil Pugalenthi and Pia Swarup**

Computational Introduction to Robotics

## Goal

The goal for this project is to control your mouse with your eyes using only a laptop webcam. We also wanted to implement a feature for mouse clicks, so we used right and left eye winking for right and left clicking respectively. This is important for accessibility reasons, such as making computer use more accessible to people with conditions like ALS, paralysis, RSI. Additionally, existing eye tracking devices are very expensive, and using a laptop webcam would be free. This also defined the constraint of our project: the built-in laptop webcam at 30 fps.

## Approach

### System overview

```
Webcam → MediaPipe → Features → Calibrated model → Smoothing → Mouse and clicks
```

The input for the model is the laptop or computer's webcam, which in our case was a 640*480 webcam at 30 frames per second. Each frame is first flipped horizontally so it works like a mirror. Then, we use Google's MediaPipe Face Landmarker on every frame to give us 478 points on the face, including the irises. MediaPipe is a machine learning model that has already been trained on extensive data sets, so we did not do any training ourselves for this landmark detection. We also get 52 blendshape scores between 0 and 1 that describe how the landmarks are moving. For example, we can see how closed each eye is, which we later use to detect winks. We also get a matrix describing the orientation of the head.

Next, we process the data we got from the MediaPipe model to 4 numbers. Look_right and look_up are found by combining the eye blendshapes. Head_yaw and head_pitch describe how far the head is turned left or right and tilted up or down. We found that it is very important to include head rotation because even when the user keeps looking at the same spot, but moves their head, the eyes rotate the other way. Therefore, the relative rotation of the eyes to the head is needed to figure out where the user is looking.

### Data collection for the model

First step to calibrate our model to the user's face, we use *calibration.py*. This is the data collecting stage for calibration. When we run, it measures the screen size and opens the webcam and starts a MediaPipe FaceLandmarker (A machine learning tool that detects the geometry of the shape and tracks different landmarks on the face such as eyes, nose, lips, etc). Then an instruction screen pops up that prompts the user through a 3 phase setup to record different instances of the face. The first phase asks the user to look at a red dot in a 3X3 grid across the screen. The second phase instructs the user to look at 5 dots generated across the screen while slowly moving the head. The third phase creates arrows that point to the screen's edges and corners and asks the user to look approximately 1 inch away from the arrow until a beep sound. For each of these targets approximately 3.5 seconds are given for the user to readjust their eyes to the target when nothing is captured. After this time frame, there is a small recording period during which a red circle shrinks around the red dot. Each camera frame recorded during the time period is stored in a row with the target's known position, whether the target was on screen or not, along with whether a face is detected. If a face was detected, then all 478 face landmarks, 52 face blendshape scores, and 16 values for the head pose matrix are recorded. When the script is done, it stores all the values in a time stamped csv file (where each frame is stored in a row) along with a JSON file that contains the metadata such as screen size, camera resolution, time settings in a calibration data folder. These files are then used to train the Gaze model which learns how to map the user's face to a screen position for the mouse.

### Gaze model

We use a quadratic regression, fitted with least squares to turn the four features into x and y coordinates on the screen. We chose a quadratic fit since it encompassed the data better than a linear relation, which did not work well when the head turned far. The model uses the data from the on screen part of the calibration and the second part of the calibration corrects for head turning.

### Smoothing

The initial prediction of the cursor location based solely on the raw iris position jittered a lot, even when the user's eyes were relatively still, so we applied some smoothing. We used a One Euro filter which smooths the motion heavily when the gaze estimation barely moves and lightly when it moves fast. Therefore, the cursor is steady when the user looks at something and still responds quickly when you look somewhere new. We also ignore changes smaller than 25 pixels and limit the top speed to 900 pixels per second.

### Mouse movement translation + Clicking

In order to translate screen gaze to mouse movement on the screen, we utilize *mouse_movement.py* which has the imported library pyautogui that consists of commands that helps translate code to mouse movement on the screen. We declared a class inside this file called `MouseMovement` which takes in parameters that consist of screen width and height along with a few tuning settings such as setting, smoothing, max speed, etc. The main method inside this class is `move_to_gaze`. This takes in parameters `gaze_x`, `gaze_y` which are the fraction of gaze points of the screen (0 - 1 which is on screen). It then stretches this point around the screen center based on the sensitivity, converts it to pixels and then treats it as a result for the cursor to move. For imitating the clicking features of the mouse, we use mediapipes left and right eye detection features. This gives a score between 0 - 1 on how open an eye is. A wink starts when one eye is 0.45 and at least 0.2 higher than the other eye. This prevents blinks as considered clicks. When a wink is held for more than 0.25 seconds, then the program returns left or right based on which eye was closed which is then translated to left or right click with commands from pyautogui.

### Important design decisions

An important design decision we had to make was what we measured the gaze relative to. We started with the midpoint between the irises in the camera image, but it moves when your head moves. So even small shifts in the head position would distort cursor movement significantly. We tried several other options such as the iris positions relative to other facial reference points like nose tip, brows, forehead, chin, temples, eye corners, and more, but none of these helped and the nose tip was especially bad since it moves more when the head moves. In the end, we use blendshapes and head rotation in a quadratic model, which improved off screen detection and reduced error with the head moving significantly.

## Challenges faced along the way

- **Transitioning from tracking the center of the irises to tracking the eyes relative to the face:** Our initial model tracked the center of the irises of the eyes and used that to draw the bounding box around where the center point could go and converted that to mouse movements. However, we quickly found out that this was inefficient. Tracking the center of the iris using the webcam produced a lot of noise which caused the cursor to jitter. Tilting the head would also ultimately cause the cursor to jump across the screen. In order to avoid this, we switched the program to track the left and right eyes with respect to other facial features to reduce cursor misinputs. We now utilize the four features and fit them to an equation from least squares regression to get x and y screen coordinates. This method helped solve our previous problem by reducing noise and erratic cursor movements.

- **Smoothing the cursor movement:** Before the cursor smoothening, the cursor would jump to the gaze locations instantly which ended up creating an unpleasant stop motion movement for the cursor. In order to combat this issue, we introduced a smoothening feature to `MouseMovement` class that would look at the current location of the mouse before moving to the new location pointed by the gaze. After, it would look at the distance between these 2 points and slow down or fasten the cursor speed based on how close or far the current cursor is to its predicted next location. This helped us get rid of the lagging cursor movements on the user's screen.

- **Winking to click:** Initially, left and right winks were confused because of image mirroring and it would click even when accidentally winking which made it annoying to use. So we changed it so it only clicks when a wink is detected for longer than 0.25s.

## Improvements with more time

- **To reduce cursor movement with head tilt:** Currently we feed in head yaw and pitch to train the model, but it doesn't take into account the sideways tilt of the head. Calibration also only takes a few head movements in a short period of time. With more time to work on the project, we would add time to calibration record more movements of the head and also take into account sideways tilt so that the cursor doesn't shift when the user moves their head.

- **Calibrate the model in a way that the user's distance from the screen doesn't cause cursor misinputs:** The current model is super dependent on the distance of the person from the camera and camera position with respect to the user. With more time, we would take calibration at two or three distances, add a distance estimate, or automatically rescale the prediction as the user moves from the camera. This will ensure that the user doesn't have to recalibrate the model every time they use the program.

- **Make the cursor snap to clickable areas on the screen when near them:** Our current webcam gaze is only accurate to a few percent within the screen. This in turn makes it harder for the user to click small buttons and links especially near the edges and corners of the screen. In order to not strain the eyes of the user, we would implement a feature where the mouse gaze snaps to the nearest clickable element when it's a certain distance away.

- **Calibrate winks for the model instead of using a constant time for better accessibility to the user:** Our current model calculates a blink when either the left or right eye is closed for 0.25 seconds. In terms of accessibility, however, it is inconvenient because the blink time varies per different people and can thus cause unintended clicks on the screen. With more time, we would include a feature in calibration that would ask the user to set a time themselves to take care of per user comfortability.

## Lessons learned

We learnt that testing, setting values like thresholds and verifying with real data is extremely important. For example, for the wink thresholds, we measured real winks instead of assuming thresholds, which worked a lot better and reduced accidental clicks a lot.

We also learnt that using visualizers to see what is going on was very useful for testing. We used bars to show the eye openness score to detect winks.

## Gen AI usage

We utilized AI in helping understand external libraries such as OpenCV and pyautogui. We also used it to generate docstrings for methods in the library as well. We also used Gen AI to debug console error messages to fix code.

## Demo

Install requirements: `pip install -r requirements.txt` and download `face_landmarker.task`

Run `python calibration.py` and then `python main.py`.

Keys: SPACE to start and pause, q to quit

Wink to click and push the mouse into a corner to to stop the program
