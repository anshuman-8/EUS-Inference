import cv2
import torch
import time
import logging as log
from PIL import Image
from model import TestLightningModule
from torchvision import transforms
import torch.nn.functional as F
from datetime import datetime, timedelta


log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[log.FileHandler("testing.log", mode="w"), log.StreamHandler()],
)

LIVE_INFERENCE = False
INFERENCE = True

FPS = 1 if LIVE_INFERENCE else 300

EUS_model_1 = "./testing/checkpoint/a7a72f80-fd9d-4f60-9cc9-1c2227375e39.ckpt"  # densenet161
EUS_model_2 = "./testing/checkpoint/35f62346-a69b-4f7d-9e5f-b384bd2f7e16.ckpt"  # resnet50
EUS_model_3 = "./testing/checkpoint/96c22c44-cb77-4347-9da3-052ef63437a0.ckpt"  # densenet161 denoising
EUS_model_4 = "./testing/checkpoint/d72fb346-f652-4605-9039-856ca4315bc2.ckpt"  # densenet161 (best)

Vid_source = "./testing/data/achamma_paulose_1967060.mkv"

# Crop details
# left, right, top, bottom
crop_params = [185, 105, 93, 90]

ground_truth = [
    {'class':'Station 1', 'start': '00:01:09', 'end': '00:08:21'},
    {'class':'Station 2', 'start': '00:08:44', 'end': '00:13:24'},
    {'class':'Station 3', 'start': '00:00:00', 'end': '00:00:00'}
]

'''
Video details:
.mkv - New Fujifilm
.asf - Old Fujifilm
.avi - Olympus

OBS Link:
Vid Src 4 - New Fujifilm

Vid Source:
video_mammen.mkv [140, 420, 105, 103] 
achamma_paulose_1967060 [0 0 0 0]

Video Inference Ground Truth:

achamma_paulose_1967060.asf
Station 1 - 00:01:09 - 00:08:21
Station 2 - 00:08:44 - 00:13:24
Station 3 - 00:00:00 - 00:00:00

video_mammen.mkv
Station 1 - 00:00:00 - 00:00:00
Station 2 - 00:00:00 - 00:00:00
Station 3 - 00:00:00 - 00:00:00
'''

cv_src = 2 if LIVE_INFERENCE else Vid_source  # 2 for HDMI port

cap = cv2.VideoCapture(cv_src, cv2.CAP_FFMPEG)

log.debug(f'Video capture fps: {cv2.CAP_PROP_FPS}')
# cap.set(FPS, 350*1000)
capture_fps = int(cap.get(cv2.CAP_PROP_FPS))
print(capture_fps)
width, height = int(cap.get(3)), int(cap.get(4))
log.info("Video capture object initialized.")

if not cap.isOpened():
    log.error("Could not access the video source.")
    exit()


IMAGE_SIZE = 224

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.22782720625400543, 0.22887665033340454, 0.23145385086536407],
            [0.11017259210348129, 0.11015155166387558, 0.11037711054086685],
        ),
    ]
)


def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    return transforms(frame)

def balanced_accuracy(y_true, y_pred):
    unique_classes = set(y_true)
    sensitivity = []

    for c in unique_classes:
        true_positive = sum((y_true == c) & (y_pred == c))
        actual_class_count = sum(y_true == c)
        sensitivity.append(true_positive / actual_class_count)

    return sum(sensitivity) / len(unique_classes)

# Convert ground truth times to seconds
for entry in ground_truth:
    entry['start'] = sum(x * int(t) for x, t in zip([3600, 60, 1], entry['start'].split(':')))
    entry['end'] = sum(x * int(t) for x, t in zip([3600, 60, 1], entry['end'].split(':')))


try:
    model = TestLightningModule.load_from_checkpoint(
        EUS_model_4, map_location="cpu"
    )
    log.debug("Model loaded from checkpoint.")
    model.eval()
    log.info("Model loaded.")
except Exception as e:
    log.error(f"Unable to load model.\n{e}")
    exit()

station_class = {0: "Station 1", 1: "Station 2", 2: "Station 3"}

correct_predictions = 0
total_predictions = 0

if INFERENCE:
    with torch.no_grad():
        start_time = time.time()
        init_time = 0
        while cap.isOpened():
            # ret = cap.grab()
            ret, frame = cap.read()
            left, right, top, bottom = crop_params
            frame = frame[top:height - bottom, left:width - right]

            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) 
            current_time = timedelta(milliseconds=current_time)
            current_time_format = f"{current_time.seconds//3600:02}:{(current_time.seconds%3600)//60:02}:{current_time.seconds%60:02}"

            if not ret:
                log.error(" Unable to capture frame.")
                break

            frame_tensor = preprocess_frame(frame)

            prediction = model(frame_tensor.unsqueeze(0)).squeeze(0).softmax(0)
            prediction_class = prediction.argmax().item()
            log.info(station_class[prediction_class])
            log.debug(f'{current_time.seconds=}')
            for entry in ground_truth:
                if entry['start'] <= current_time.total_seconds() <= entry['end']:
                    if station_class[prediction_class] == entry['class']:
                        log.info(f"Correct prediction: {station_class[prediction_class]}")
                        correct_predictions += 1
                    else:
                        log.info(f"Incorrect prediction: {station_class[prediction_class]}")
                    total_predictions += 1

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
            cv2.putText(
                    frame,
                    current_time_format,
                    (width - 370, height - 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("EUS Inference", frame)

            elapsed_time = time.time() - start_time

            init_time += elapsed_time

            cap.set(cv2.CAP_PROP_POS_MSEC, (init_time) * 1000 )
            log.debug(f'moving to {elapsed_time}')
            start_time = start_time + elapsed_time

            if cv2.waitKey(capture_fps) & 0xFF == ord("q"):
                log.info("Exiting.")
                break
else:
    while cap.isOpened():
        ret = cap.grab()
        ret, frame = cap.retrieve()
        left, right, top, bottom = crop_params
        frame = frame[top:height - bottom, left:width - right]

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) 
        current_time = timedelta(milliseconds=current_time)
        current_time_format = f"{current_time.seconds//3600:02}:{(current_time.seconds%3600)//60:02}:{current_time.seconds%60:02}"

        cv2.putText(
                    frame,
                    str(current_time_format),
                    (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if not ret:
            log.error(" Unable to capture frame.")
            break
    
        cv2.imshow("EUS Inference", frame)
        if cv2.waitKey(FPS) & 0xFF == ord("q"):
            log.info("Exiting.")
            break 

cap.release()
cv2.destroyAllWindows()
