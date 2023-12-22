import sys
import json
import torch
import logging as log
from PyQt5.QtWidgets import QApplication

from src.main import App


def main():
    log.basicConfig(
        level=log.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[log.FileHandler("Inference-running.log", mode="w"), log.StreamHandler()],
    )

    # Load settings from settings.json
    try:
        with open("settings.json", "r") as json_file:
            inference_config = json.load(json_file)
            log.info("Settings loaded from settings.json")
    except Exception as e:
        log.error(f"Error loading settings from settings.json: {e}")
        sys.exit(1)

    expected_keys = [
        "checkpoint_path",
        "device",
        "image_size",
        "mean",
        "std",
        "station_class",
        "video_source",
        "window_width",
        "window_height",
        "display_width",
        "display_height",
    ]

    for key in expected_keys:
        if key not in inference_config:
            if key == "video_source":
                inference_config[key] = 2
            else:
                log.error(f"Missing key '{key}' in settings.json")
                sys.exit(1)

    device = torch.device(
        inference_config["device"] if torch.cuda.is_available() else "cpu"
    )
    log.info(f"Device: {device}")

    app = QApplication(sys.argv)
    a = App(config=inference_config)
    a.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
