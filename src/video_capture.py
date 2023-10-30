import sys
import cv2
import numpy as np
import logging as log
from PyQt5.QtCore import pyqtSignal, QThread

from src.Inference.main import Inference
from src.Inference.model import TestLightningModule
from src.Inference.utils import isDisturbed, preprocess_frame


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    no_video_signal = pyqtSignal()
    prediction_signal = pyqtSignal(str)

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
                frame, station = self.EUSInference.perform_inference_on_frame(frame)
                if station is not None:
                    self.prediction_signal.emit(station)
                else:
                    self.prediction_signal.emit('-')
            
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
