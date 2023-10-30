import torch
import cv2
import logging as log
from src.Inference.model import TestLightningModule
from src.Inference.utils import isDisturbed, preprocess_frame

class Inference:
    def __init__(self, model, device="cuda"):
        # init model
        self.model_path = model
        self.device = 'cpu'
        self.model = None
        try:
            self.model = TestLightningModule.load_from_checkpoint(
                self.model_path, map_location=self.device
            )
            log.debug("Model loaded from checkpoint.")
            self.model.eval()
            log.info("Model loaded.")
        except Exception as e:
            log.error(f"Unable to load model.\n{e}")
            raise e
        
        self.station_class = {0: '1', 1: '2', 2: '3'}

    def perform_inference_on_frame(self, frame):
        # check if image disturbed
        disturbance, reason = isDisturbed(frame)

        if disturbance:
            cv2.putText(
                frame,
                reason,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
            log.info("Image is disturbed reason: " + reason)
            
            return frame, None
        
        # preprocess the frame
        frame_tensor = preprocess_frame(frame)

        # perform inference
        prediction = self.model(frame_tensor.unsqueeze(0)).squeeze(0).softmax(0)
        prediction_class = prediction.argmax().item()
        log.info(f': Station {self.station_class[prediction_class]}')

        # paste inference results
        for i in range(3):
            text = f"Station {self.station_class[i]}: {prediction[i].item()*100:.2f}%"
            position = (10, (i + 1) * 60)
            color = (255, 255, 255)
            cv2.putText(
                frame,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                2.2,
                color,
                3,
                cv2.LINE_AA,
            )

        return frame, self.station_class[prediction_class]