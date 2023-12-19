import cv2
import numpy as np
import logging as log
from typing import Dict, Any, List
from PyQt5.QtCore import pyqtSignal, QThread

from src.Inference.main import Inference
from src.Inference.utils import crop_frame


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    no_video_signal = pyqtSignal()
    station_signal = pyqtSignal(str)
    predictions_signal = pyqtSignal(list)

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self._run_flag = True
        self.inference = False
        self.EUSInference = None
        self.config = config
        self.crop_dim = config.get("crop_dim", [0,0,0,0])

    def run(self):
        """Capture video from camera and send frames to main thread using signals."""
        video_source = self.config.get('video_source', 2)
        cap = cv2.VideoCapture(video_source)
        while self._run_flag:
            ret, frame = cap.read()

            if not ret:
                log.error(" Unable to capture frame.")
                self.no_video_signal.emit() # Emit signal when no video input
                self.station_signal.emit('-')
                break

            # crop frame 
            frame = crop_frame(frame, self.crop_dim)

            if self.inference:
                frame, predictions = self.EUSInference.perform_inference_on_frame(frame)
                if predictions is not None and self.inference:
                    station = predictions.index(max(predictions)) + 1
                    prediction = f'Station: {station}'
                    self.station_signal.emit(prediction)
                    self.predictions_signal.emit(predictions)
                else:
                    def_station = [0.0, 0.0, 0.0]
                    self.station_signal.emit('-')
                    self.predictions_signal.emit(def_station)
            
            self.change_pixmap_signal.emit(frame)
            
        cap.release()

    def start_inference(self):
        """Sets inference flag to True and loads model"""
        try:
            self.EUSInference = Inference(config = self.config)
        except Exception as e:
            log.error(f"Unable to load model.\n{e}")
            self.stop()
            exit(1)

        self.inference = True

    def stop_inference(self):
        """Sets inference flag to False"""
        self.inference = False
        self.EUSInference = None
        log.debug("Inference model stopped")
        self.station_signal.emit('-')


    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False

        self.wait()
