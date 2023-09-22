import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Initialize the video capture object
cap = cv2.VideoCapture("../data/video_mammen.mkv")  # 0 for default webcam. Change to a file path for a video file.

# Check if the camera/video source is opened successfully
if not cap.isOpened():
    print("Error: Could not access the video source.")
    exit()

# Loop to continuously capture frames
while True:
    # Capture a frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Display the frame
    cv2.imshow('Live Video Feed', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
