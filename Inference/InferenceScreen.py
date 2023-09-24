import sys
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QProgressBar, QWidget, QLabel, QVBoxLayout, QPushButton, QFrame

class InferenceScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.station1_value = 0
        self.station2_value = 0
        self.station3_value = 0
        self.render()

    def render(self):
        layout = QVBoxLayout()

        # Video Feed Frame
        video_frame = QFrame(self)
        video_frame.setFrameShape(QFrame.Shape.Box)
        video_frame.setFrameShadow(QFrame.Shadow.Plain)
        video_frame.setLineWidth(2)
        video_frame.setMidLineWidth(1)
        video_layout = QVBoxLayout()

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

         # Add Station Labels and Progress Bars
        stations = {0:"Station 1", 1:"Station 2", 2:"Station 3"}
        self.progress_bars = []

        for station in stations.keys():
            label = QLabel(f"{stations[station]} : ")
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            self.progress_bars.append(progress_bar)

            h_layout = QHBoxLayout()
            h_layout.addWidget(label)
            h_layout.addWidget(progress_bar)
            layout.addLayout(h_layout)


        stop_button = QPushButton('Stop')
        stop_button.clicked.connect(self.showFirstScreen)
        layout.addWidget(stop_button)

        self.setLayout(layout)

    def updateProgressBar(self, station, value):
        self.progress_bars[station].setValue(value)


    def showFirstScreen(self):
        self.parent().setCurrentIndex(0)