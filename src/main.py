import sys
import cv2
import numpy as np
import logging as log
from PyQt5 import QtGui
from typing import Dict, Any
from PyQt5.QtCore import pyqtSlot, Qt
from src.video_capture import VideoThread
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QVBoxLayout, QHBoxLayout


class App(QWidget):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # App config
        self.config = config

        # Window init
        self.setWindowTitle("EUS - ML")
        self.setFixedSize(config['window_width'], config['window_height'])

        self.disply_width = config['display_width']
        self.display_height = config['display_height']

        # Label that holds the image
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.disply_width, self.display_height)  # Set fixed label size


        # create an empty placeholder pixmap
        self.placeholder_pixmap = QPixmap(self.disply_width, self.display_height)
        self.placeholder_pixmap.fill(Qt.gray) 

        # If no Inference
        painter = QPainter(self.placeholder_pixmap)
        font = painter.font()
        font.setPointSize(20)
        painter.setFont(font)
        painter.drawText(self.disply_width//4, self.display_height//2, "No video source found :(")

        # create a flag to track if video input is available
        self.video_available = True

        # create the video capture thread
        self.thread = VideoThread(config=config)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start() #start the thread

        # connect the no_video_signal to the update_no_video method
        self.thread.no_video_signal.connect(self.update_no_video)

        # Text label
        self.textLabel = QLabel('EUS Prediction')
        self.textLabel.setStyleSheet(
            "QLabel {"
            "color: white;"
            "font-size: 30px;"
            "}"
        )

        self.error_message_box = QLabel('')
        self.error_message_box.setStyleSheet(
            "QLabel {"
            "color: red;"  
            "font-size: 18px;"
            "}"
        )
        self.error_message_box.setAlignment(Qt.AlignCenter)
        self.error_message_box.setFixedSize(self.disply_width, 30)

        # Inference block
        # Vertical box layout 
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.image_label, alignment=Qt.AlignCenter)  
        vbox.addWidget(self.textLabel, alignment=Qt.AlignCenter)
        
        vbox.addStretch(1)
        # vbox.addWidget(self.error_message_box, alignment=Qt.AlignCenter)

        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # Background color
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(50, 50, 60))  # Dark gray background
        self.setPalette(palette)

        #Start button
        self.start_button = QPushButton('Start Prediction ', self)
        self.start_button.setFixedSize(200, 50)
        self.setButtonStyle(self.start_button)
        self.start_button.clicked.connect(self.toggle_inference)

        # Refresh button
        self.refresh_button = QPushButton('Refresh', self)
        vbox.addWidget(self.refresh_button, alignment=Qt.AlignCenter)
        self.setButtonStyle(self.refresh_button)
        self.refresh_button.clicked.connect(self.refresh_video_source)

        # Horizontal layout for buttons and inference results
        hbox = QHBoxLayout()

        controlv = QVBoxLayout()
        controlv.addWidget(self.start_button, alignment=Qt.AlignCenter)
        controlv.addWidget(self.refresh_button, alignment=Qt.AlignCenter)
        controlv.addStretch(1)  # Add spacing between buttons and results

        # Placeholder for inference results
        self.inference_result = QLabel('', self)
        self.thread.prediction_signal.connect(self.update_inference_result)
        self.inference_result.setAlignment(Qt.AlignCenter)
        self.inference_result.setStyleSheet(
            "QLabel {"
            "color: white;"
            "font-size: 35px;"
            "}"
        )
        self.inference_result.setFixedSize(200, 45)

        hbox.addLayout(controlv)
        hbox.addWidget(self.inference_result, alignment=Qt.AlignCenter)

        vbox.addLayout(hbox)

        self.inference_running = False

    def setButtonStyle(self, widget):
        """Set common style for buttons"""
        widget.setFixedSize(200, 45)
        widget.setStyleSheet(
            "QPushButton {"
            "background-color: #4CAF50;"  # Green background
            "border: none;"
            "color: white;"
            "padding: 10px 20px;"
            "text-align: center;"
            "text-decoration: none;"
            "font-size: 16px;"
            "margin: 4px 2px;"
            "border-radius: 10px;"
            "}"
            "QPushButton:hover {background-color: #45a049;}"  # Darker green on hover
            "QPushButton:pressed {background-color: #FF5733;}"  # Red when pressed
        )
    
    def update_image_border(self):
        """Updates the frame border based on the inference state"""
        if self.inference_running:
            self.image_label.setStyleSheet("border: 2px solid green;")
        else:
            self.image_label.setStyleSheet("border: 0px;")

    def closeEvent(self, event):
        """Shuts down the thread on app close"""
        self.thread.stop()
        event.accept()

    def refresh_video_source(self):
        """Refresh the video source"""
        # Stop the current thread
        self.thread.stop()

        # Start a new thread to capture video
        self.thread = VideoThread(config=self.config)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.no_video_signal.connect(self.update_no_video)
        self.thread.start()

    def toggle_inference(self):
        """To Start and Stop the inference """
        # Perform model inference on frames
        if self.video_available is False:
            self.textLabel.setText('No video source found ')
            return
        
        if self.thread.inference:
            self.textLabel.setText('Stopping ... ')
            self.thread.stop_inference()
            self.textLabel.setText('EUSML Inference Stopped')
            self.start_button.setText('Start Inference')
            self.inference_running = False
            self.update_image_border()
            return
        
        self.textLabel.setText('Starting ... ')
        self.textLabel.setText('EUSML Inference Running')  # Update label text
        log.debug(f"Starting inference using checkpoint {self.config['checkpoint_path']} ")
        self.thread.start_inference()  # Start inference  
        self.thread.inference = True
        self.start_button.setText('Stop Inference')
        self.inference_running = True
        self.update_image_border()

    @pyqtSlot()
    def update_no_video(self):
        """Updates the image_label with a placeholder pixmap when no video input"""

        log.info("shifting to Video placeholder ")
        if self.video_available:

            if self.thread.inference:
                self.toggle_inference()
                log.info("Inference stopped")

            self.image_label.setPixmap(self.placeholder_pixmap)
            painter = QPainter(self.placeholder_pixmap)
            font = painter.font()
            font.setPointSize(20)
            painter.setFont(font)
            painter.drawText(self.disply_width//4, self.display_height//2, "No video source")
            self.video_available = False
            self.inference_result.setText('-')

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)

        # Set the frame border color based on the inference state
        if self.inference_running:
            self.image_label.setStyleSheet("border: 2px solid green;")
        else:
            self.image_label.setStyleSheet("border: 0px;")

        self.image_label.setPixmap(qt_img)
        self.video_available = True  # Set video_available flag to True

    @pyqtSlot(str)
    def update_inference_result(self, prediction):
        """Updates the Station prediction label"""
        if self.video_available is False:
            log.debug("No prediction as no video source found")
            self.inference_result.setText(f'')
            return
        self.inference_result.setText(f'Station: {prediction}')
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
