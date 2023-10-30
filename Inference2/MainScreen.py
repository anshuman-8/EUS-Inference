from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QFrame,
    QVBoxLayout,
    QPushButton,
    QComboBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage
import cv2
import logging as log

class MainScreen(QWidget):
    def __init__(self, video_source=2):
        super().__init__()
        self.video_source = video_source
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Title Label
        title_label = QLabel('EUS')
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = title_label.font()
        title_font.setPointSize(24)  # Increase font size
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Video Feed Frame
        video_frame = QFrame(self)
        video_frame.setFrameShape(QFrame.Shape.Box)
        video_frame.setFrameShadow(QFrame.Shadow.Plain)
        video_frame.setLineWidth(2)
        video_frame.setMidLineWidth(1)
        video_layout = QVBoxLayout()

        self.cap = cv2.VideoCapture(self.video_source)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Video Label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.video_label)

        # No Video Label
        self.no_video_label = QLabel('No video input found')
        self.no_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_video_label.setVisible(False)
        video_layout.addWidget(self.no_video_label)

        if not self.cap.isOpened():
            self.no_video_label.setVisible(True)
            self.video_label.setVisible(False)
        else:
            self.no_video_label.setVisible(False)
            self.video_label.setVisible(True)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.updateVideoFrame)
            self.timer.start(30)  # Adjust the timeout interval as needed

        video_frame.setLayout(video_layout)
        layout.addWidget(video_frame)

        # Model Dropdown
        model_dropdown = QComboBox()
        model_dropdown.addItems(["Model 1", "Model 2", "Model 3"])
        layout.addWidget(model_dropdown)

        # Video Source Dropdown
        vid_dropdown = QComboBox()
        vid_dropdown.addItems(["Vid 0", "Vid 1", "Vid 2"])
        vid_dropdown.currentIndexChanged.connect(self.updateVideoSource)
        layout.addWidget(vid_dropdown)

        # Start Button
        start_button = QPushButton('Start')
        start_button.clicked.connect(self.showSecondScreen)
        start_button.setFixedSize(200, 50)  # Set a fixed size for the button
        layout.addWidget(start_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)

    def updateVideoFrame(self):
        ret, frame = self.cap.read()

        if ret:
            # Convert frame to QImage
            frame = cv2.resize(frame, (1000, 700))
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()

            # Set QImage as pixmap for QLabel
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap)
            self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video_label.setScaledContents(True)

    def updateVideoSource(self, index):
        self.cap.release()
        self.parent().setVideoSource(index)
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            self.no_video_label.setVisible(True)
            self.video_label.setVisible(False)
            log.error(f"Error: Could not access video source {self.video_source}")
        else:
            self.no_video_label.setVisible(False)
            self.video_label.setVisible(True)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.updateVideoFrame)
            self.timer.start(30)

    def showSecondScreen(self):
        self.cap.release()
        self.parent().setCurrentIndex(1)