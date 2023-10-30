import torch
import logging as log
from PIL import Image
from torchvision import transforms
from Inference.model import LTModule


class Inference():
    def __init__(
        self,
        model_path,
        mean,
        std,
        device="cpu",
        image_size=224,
        station_class={0: "Station 1", 1: "Station 2", 2: "Station 3"},
    ):
        self.model = None
        self.device = device
        self.model_path = model_path
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.station_class = station_class
        self.transforms = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.load_model()
        

    def load_model(self):
        try:
            self.model = LTModule.load_from_checkpoint(
                self.model_path, map_location=self.device
            )
            self.model.eval()
            log.info("Model loaded.")
        except Exception as e:
            log.error(f"Unable to load model.\n{e}")
            exit()


    def preprocess_frame(self, frame):
        frame = Image.fromarray(frame)
        return self.transforms(frame)


    def predict(self, frame):
        frame = self.preprocess_frame(frame).to(self.device)
        with torch.no_grad():
            prediction = self.model(frame.unsqueeze(0)).squeeze(0).softmax(0)
            prediction_class = prediction.argmax().item()
            log.info(self.station_class[prediction_class])

            return prediction_class
