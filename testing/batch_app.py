
# import lightning as L


# class YourComponent(L.LightningWork):
#    def run(self):
#       print('RUN ANY PYTHON CODE HERE')


# component = YourComponent()
# app = L.LightningApp(component)


import os
import cv2
import numpy as np
import torch
import logging as log
from PIL import Image
from model_arc import Discriminator
from torchvision import transforms
import torch.nn.functional as F

model_path = "testing/model/face-disc-test-1.pth"

# Initialize the video capture object
cap = cv2.VideoCapture(0)  
log.info("Video capture object initialized.")

if not cap.isOpened():
    log.error("Error: Could not access the video source.")
    exit()


IMAGE_SIZE = 128
CHANNELS_IMG = 3
BATCH_SIZE = 16
FEATURES = 128

transforms = transforms.Compose(
    [  
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
    ]
)

def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    return transforms(frame)

model = Discriminator(CHANNELS_IMG, FEATURES)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
log.info("Model loaded.")

batch = torch.empty((0, 3, IMAGE_SIZE, IMAGE_SIZE))
# batch = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame.")
        break
    
    frame_tensor = preprocess_frame(frame)
    log.info("Frame preprocessed.")
   
    batch = torch.cat([batch, frame_tensor.unsqueeze(0)],dim=0 )
    log.info("Frame added to batch.")

    if batch.shape[0] >= 16:

      #   batch_tensor = torch.tensor(batch)

        with torch.no_grad():
            output = model(batch)

        print(output)
        class_predictions = torch.argmax(output, dim=1)
        class_probabilities = F.softmax(output, dim=1)

        # Display the frames with classification details
        for i, prediction in enumerate(class_predictions):
            classification_percentage = class_probabilities[i][prediction].item() * 100

            # Overlay classification details on the frame
            cv2.putText(frame, f'Class {prediction}: {classification_percentage:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('Live Video Feed', frame)

            # Clear the batch
            batch = []

       # Display the frame
        cv2.imshow('Live Video Feed', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
