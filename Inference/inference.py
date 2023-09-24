import cv2
import torch
import logging as log
from PIL import Image
from torchvision import transforms


class Inference:
    def __init__(self, model_path):
        self.model = None
        self.image_size = 224
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.station_class = {0: "Station 1", 1: "Station 2", 2: "Station 3"}
        self.transforms = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.22782720625400543, 0.22887665033340454, 0.23145385086536407],
                    [0.11017259210348129, 0.11015155166387558, 0.11037711054086685],
                ),
            ]
        )
        pass

    def preprocess_frame(self, frame):
        frame = Image.fromarray(frame)
        return transforms(frame)

    def predict(self, frame):
        with torch.no_grad():
            prediction = self.model(frame.unsqueeze(0)).squeeze(0).softmax(0)
            prediction_class = prediction.argmax().item()
            log.info(self.station_class[prediction_class])

            return prediction_class
