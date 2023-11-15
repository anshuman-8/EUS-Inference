import torch
import sys
import json
from PyQt5.QtWidgets import QApplication
import logging as log
from src.main import App

def main():
    log.basicConfig(
        level=log.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[log.FileHandler("testing.log", mode="w"), log.StreamHandler()],
    )

    # Load settings from settings.json
    try:
        with open('settings.json', 'r') as json_file:
            inference_config = json.load(json_file)
    except Exception as e:
        log.error(f"Error loading settings from settings.json: {e}")
        sys.exit(1)

    device = torch.device(inference_config["device"] if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    app = QApplication(sys.argv)
    a = App(config = inference_config)
    a.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
