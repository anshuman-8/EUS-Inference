import sys
import cv2
import numpy as np
import logging as log
from PyQt5.QtCore import pyqtSignal, QThread
from Inference.model import TestLightningModule
from Inference.utils import isDisturbed, preprocess_frame



class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    no_video_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.inference = False
        self.EUSInference = None

    def run(self):
        cap = cv2.VideoCapture(2)
        while self._run_flag:
            ret, frame = cap.read()

            if not ret:
                log.error(" Unable to capture frame.")
                self.no_video_signal.emit()  # Emit signal when no video input
                break

            new_width, new_height = frame.shape[1], frame.shape[0]

            if self.inference:
                frame = self.EUSInference.perform_inference_on_frame(frame)
            
            self.change_pixmap_signal.emit(frame)
            
        cap.release()

    def start_inference(self, model_path):
        try:
            self.EUSInference = Inference(model=model_path)
        except Exception as e:
            log.error(f"Unable to load model.\n{e}")
            self.stop()
            exit()

        self.inference = True

    def stop_inference(self):
        self.inference = False
        self.EUSInference = None


    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


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
            log.info("Image is disturbed reason: " + reason)
            return frame
        
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

        return frame  