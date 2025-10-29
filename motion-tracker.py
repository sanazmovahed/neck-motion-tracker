"""
PhD Applicant Task - Skin/Neck Motion Tracking
Author: Sanaz Movahed
Method: Optical Flow (Farnebäck) via OpenCV
Objective: Visualize subtle skin motion in neck region using RGB video
"""


import cv2
import numpy as np
import os

# ---------- SETTINGS ----------
VIDEO_PATH = "NeckDeformationClip.mp4"   # Video Clip's Path
OUTPUT_PATH = "neck_motion_output_tracker.mp4"   # Path of output file
DISPLAY_SCALE = 1.0                     
SAVE_VIDEO = True                  
# ------------------------------

# Opening the video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError("Video is not valid!")

# Getting video's info
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set videoWriter to output video
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

# Getting first frame and convert it to grayscale
ret, prev_frame = cap.read()
if not ret:
    raise ValueError("Error in getting first frame")

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

frame_count = 0

print("Optical Flow tracking started... Press ESC to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Processing is DONE!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Getting magnitudes and angles
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Making HSV video to show motion with colors and brightness
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255

    # Hue = angle, Value = motion intensity
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    motion_color = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    combined = np.hstack((frame, motion_color))

    if DISPLAY_SCALE != 1.0:
        combined = cv2.resize(combined, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    cv2.imshow("Neck Optical Flow (ESC to exit)", combined)

    # Saving in output file
    if SAVE_VIDEO:
        out.write(motion_color)

    # Closing with ESC button
    if cv2.waitKey(1) & 0xFF == 27:
        break

    prev_gray = gray.copy()
    frame_count += 1

cap.release()
if SAVE_VIDEO:
    out.release()
cv2.destroyAllWindows()

print(f"Video is saved as: {OUTPUT_PATH}")
print(f"Total processed Frames: {frame_count}")

