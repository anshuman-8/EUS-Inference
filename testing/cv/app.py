import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # 0 for default webcam. Change to a file path for a video file.

# Check if the camera/video source is opened successfully
if not cap.isOpened():
    print("Error: Could not access the video source.")
    exit()
# Define a function to preprocess frames
def preprocess_frame(frame):
    # Preprocess frame as needed (e.g., resizing, normalization)
    # ...

    # return preprocessed_frame
    pass

# model = torch.load('model.pth')
# model.eval()

batch = []
# Loop to continuously capture frames
while True:
    # Capture a frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Unable to capture frame.")
        break
    print(ret)
    
    # preprocessed_frame = preprocess_frame(frame)

    # batch.append(preprocessed_frame)

    # if len(batch) == 16:
    #     # Convert the batch to a tensor
    #     batch_tensor = torch.tensor(batch)

    #     # Make prediction
    #     with torch.no_grad():
    #         output = model(batch_tensor)

    #     # Assuming 'output' is a tensor with classification scores
    #     # Process the output to get class predictions and percentages
    #     class_predictions = torch.argmax(output, dim=1)
    #     class_probabilities = F.softmax(output, dim=1)

    #     # Display the frames with classification details
    #     for i, prediction in enumerate(class_predictions):
    #         classification_percentage = class_probabilities[i][prediction].item() * 100

    #         # Overlay classification details on the frame
    #         cv2.putText(frame, f'Class {prediction}: {classification_percentage:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    #         # Display the frame
    #         cv2.imshow('Live Video Feed', frame)

    #         # Clear the batch
    #         batch = []

    # Display the frame
    cv2.imshow('Live Video Feed', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
