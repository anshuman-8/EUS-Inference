import sys
import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, Qt
from src.video_capture import VideoThread
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout


class App(QWidget):
    def __init__(self, model:str, vid_src:int = 2):
        super().__init__()
        # Inference details
        self.model_path = model
        self.vid_src = vid_src

        # Window init
        self.setWindowTitle("EUS - ML")
        self.setFixedSize(800, 800)
        self.disply_width = 640
        self.display_height = 640

        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.disply_width, self.display_height)  # Set fixed label size

        # create a text label
        self.textLabel = QLabel('EUSML Inference')
        self.textLabel.setStyleSheet(
            "QLabel {"
            "color: white;"
            "font-size: 20px;"
            "}"
        )

        # Inference block
        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.image_label, alignment=Qt.AlignCenter)  # Align label to center
        vbox.addWidget(self.textLabel, alignment=Qt.AlignCenter)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # Set a background color for the widget
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(50, 50, 50))  # Dark gray background
        self.setPalette(palette)

        # create an empty placeholder pixmap
        self.placeholder_pixmap = QPixmap(self.disply_width, self.display_height)
        self.placeholder_pixmap.fill(Qt.gray) 

        # If no Inference
        painter = QPainter(self.placeholder_pixmap)
        font = painter.font()
        font.setPointSize(20)
        painter.setFont(font)
        painter.drawText(self.disply_width//4, self.display_height//2, "No video source")

        # create a flag to track if video input is available
        self.video_available = True

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

        # connect the no_video_signal to the update_no_video method
        self.thread.no_video_signal.connect(self.update_no_video)

        self.start_button = QPushButton('Start Inference ', self)
        self.start_button.setFixedSize(200, 45)
        self.start_button.setStyleSheet(
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
        self.start_button.clicked.connect(self.toggle_inference)

        # Create a refresh button
        self.refresh_button = QPushButton('Refresh', self)
        self.refresh_button.clicked.connect(self.refresh_video_source)
        vbox.addWidget(self.refresh_button, alignment=Qt.AlignCenter)

        self.setStyle(self.refresh_button)

        # Horizontal layout for buttons and inference results
        hbox = QHBoxLayout()
        hbox.addWidget(self.start_button, alignment=Qt.AlignCenter)
        hbox.addWidget(self.refresh_button, alignment=Qt.AlignCenter)
        hbox.addStretch(1)  # Add spacing between buttons and results

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
        hbox.addWidget(self.inference_result, alignment=Qt.AlignCenter)

        vbox.addLayout(hbox)

    def setStyle(self, widget):
        # Set common style for buttons
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
        )

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def refresh_video_source(self):
        # Stop the current thread
        self.thread.stop()

        # Start a new thread to capture video
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.no_video_signal.connect(self.update_no_video)
        self.thread.start()

    def toggle_inference(self):
        # Perform model inference on frames
        if self.thread.inference:
            self.textLabel.setText('Stopping ... ')
            self.thread.stop_inference()
            self.textLabel.setText('EUSML Inference Stopped')
            self.start_button.setText('Start Inference')
            return
        
        self.textLabel.setText('Starting ... ')
        self.textLabel.setText('EUSML Inference Running')  # Update label text
        self.thread.start_inference(model_path=self.model_path)  # Start inference  
        self.start_button.setText('Stop Inference')

    @pyqtSlot()
    def update_no_video(self):
        """Updates the image_label with a placeholder pixmap when no video input"""
        if self.video_available:
            self.image_label.setPixmap(self.placeholder_pixmap)
            painter = QPainter(self.placeholder_pixmap)
            font = painter.font()
            font.setPointSize(20)
            painter.setFont(font)
            painter.drawText(self.disply_width//4, self.display_height//2, "No video source")
            self.video_available = False

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        self.video_available = True  # Set video_available flag to True

    @pyqtSlot(str)
    def update_inference_result(self, prediction):
        self.inference_result.setText(f'Station: {prediction}')
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
