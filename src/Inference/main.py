import torch
import cv2
import logging as log
from typing import Dict, Any
from src.Inference.model import TestLightningModule
from src.Inference.utils import isDisturbed, preprocess_frame, crop_frame

class Inference:
    def __init__(self, config: Dict[str, Any]):
        # init model
        self.model_path = config["checkpoint_path"]
        self.device = config["device"] if torch.cuda.is_available() else "cpu"
        self.model = None
        self.station_class = {0: '1', 1: '2', 2: '3'}
        self.config = config

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
        

    def perform_inference_on_frame(self, frame):
        """Performs inference on a single frame"""
        disturbance, reason = isDisturbed(frame)

        if disturbance:
            cv2.putText(
                frame,
                reason,
                (4, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            log.info("Image is disturbed reason: " + reason)
            
            return frame, None
        
        # preprocess the frame
        frame_tensor = preprocess_frame(frame, self.config["image_size"], self.config["mean"], self.config["std"])

        # perform inference
        prediction = self.model(frame_tensor.unsqueeze(0)).squeeze(0).softmax(0)
        prediction_class = prediction.argmax().item()
        log.info(f': Station {self.station_class[prediction_class]}')

        # paste inference results
        for i in range(3):
            text = f"Station {self.station_class[i]}: {prediction[i].item()*100:.2f}%"
            position = (4, (i + 1) * 20)
            color = (255, 255, 255)
            cv2.putText(
                frame,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                1,
                cv2.LINE_AA,
            )

        return frame, self.station_class[prediction_class]