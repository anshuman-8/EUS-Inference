import cv2
import torch
import timm
import logging as log
from PIL import Image
from model import TestLightningModule
from torchvision import transforms
import torch.nn.functional as F

log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[log.FileHandler("testing.log", mode="w"), log.StreamHandler()],
)

EUS_model_1 = "./checkpoint/a7a72f80-fd9d-4f60-9cc9-1c2227375e39.ckpt"  # densenet161
EUS_model_2 = "./checkpoint/35f62346-a69b-4f7d-9e5f-b384bd2f7e16.ckpt"  # resnet50
EUS_model_3 = "./checkpoint/96c22c44-cb77-4347-9da3-052ef63437a0.ckpt"  # densenet161 denoising
EUS_model_4 = "./checkpoint/d72fb346-f652-4605-9039-856ca4315bc2.ckpt"  # densenet161 gaussian smoothing

cap = cv2.VideoCapture(2)  # 2 for HDMI port
cap.set(cv2.CAP_PROP_FPS, 30)
log.info("Video capture object initialized.")

if not cap.isOpened():
    log.error("Could not access the video source.")
    exit()


IMAGE_SIZE = 224

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        #   transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.22782720625400543, 0.22887665033340454, 0.23145385086536407],
            [0.11017259210348129, 0.11015155166387558, 0.11037711054086685],
        ),
    ]
)


def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    return transforms(frame)


try:
    model = TestLightningModule(model=None)
    model.load_from_checkpoint(
        EUS_model_4, map_location="cpu"
    )
    log.debug("Model loaded from checkpoint.")
    model.eval()
    log.info("Model loaded.")
except Exception as e:
    log.error(f"Unable to load model.\n{e}")
    exit()

station_class = {0: "Station 1", 1: "Station 2", 2: "Station 3"}

while True:
    ret, frame = cap.read()

    if not ret:
        log.error(" Unable to capture frame.")
        break

    frame_tensor = preprocess_frame(frame)

    with torch.no_grad():
        prediction = model(frame_tensor.unsqueeze(0)).squeeze(0).softmax(0)
        prediction_class = prediction.argmax().item()
        log.info(station_class[prediction_class])

        for i in range(3):
            text = f"{station_class[i]}: {prediction[i].item()*100:.2f}%"
            position = (10, (i + 1) * 32)
            color = (255, 255, 255)
            cv2.putText(
                frame,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
                cv2.LINE_AA,
            )

    cv2.imshow("EUS Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        log.info("Exiting.")
        break

cap.release()
cv2.destroyAllWindows()
