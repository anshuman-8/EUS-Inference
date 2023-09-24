from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QStackedWidget,
    QComboBox,
)
from PyQt6.QtCore import Qt, QTimer
from Inference.InferenceScreen import InferenceScreen
from Inference.MainScreen import MainScreen


class MainApplication(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.stackedWidget = QStackedWidget()

        first_screen = MainScreen()
        first_screen.setParent(self)
        self.stackedWidget.addWidget(first_screen)

        second_screen = InferenceScreen()
        second_screen.setParent(self)
        self.stackedWidget.addWidget(second_screen)

        self.setCentralWidget(self.stackedWidget)

        self.setWindowTitle('EUS Application')
        self.setGeometry(500, 500, 500, 300)

        self.show()

