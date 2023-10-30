import torch
import sys
from PyQt5.QtWidgets import  QApplication
import logging as log
from src.main import App

def main():
    log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[log.FileHandler("testing.log", mode="w"), log.StreamHandler()],
    )

    # log.info("Video capture object initialized.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    inference_config = {
        'checkpoint_path': 'checkpoints/d72fb346-f652-4605-9039-856ca4315bc2.ckpt',
        'device':  device,
        'image_size': 224,
        'mean': [0.22782720625400543, 0.22887665033340454, 0.23145385086536407],
        'std': [0.11017259210348129, 0.11015155166387558, 0.11037711054086685],
        'station_class': {0: "Station 1", 1: "Station 2", 2: "Station 3"},
        'log_format': '%(asctime)s %(levelname)s %(message)s',
        'video_source': 2,
        'fps': 30,
        'frame_timeout': 1,
    }

    app = QApplication(sys.argv)
    a = App("./checkpoints/d72fb346-f652-4605-9039-856ca4315bc2.ckpt")
    a.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
