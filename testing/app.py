import cv2
import torch
import time
import logging as log
import numpy as np
from PIL import Image
from model import TestLightningModule
from torchvision import transforms
import torch.nn.functional as F
from datetime import timedelta
from sklearn.metrics import balanced_accuracy_score



log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[log.FileHandler("testing.log", mode="w"), log.StreamHandler()],
)

EUS_model_1 = "./testing/checkpoint/a7a72f80-fd9d-4f60-9cc9-1c2227375e39.ckpt"  # densenet161
EUS_model_2 = "./testing/checkpoint/35f62346-a69b-4f7d-9e5f-b384bd2f7e16.ckpt"  # resnet50
EUS_model_3 = "./testing/checkpoint/96c22c44-cb77-4347-9da3-052ef63437a0.ckpt"  # densenet161 denoising
EUS_model_4 = "./testing/checkpoint/d72fb346-f652-4605-9039-856ca4315bc2.ckpt"  # densenet161 (best)

Vid_source = "./testing/data/achamma_paulose_1967060.mkv"

'''
Video details:

OBS Link:
Vid Src 4 - New Fujifilm

Video Inference Ground Truth:

achamma_paulose_1967060.asf (Old Olympus)
src - ./testing/data/achamma_paulose_1967060.mkv
crop_params = [185, 105, 93, 90] 
trained - 
test - 
Station 1 - 00:01:09 - 00:08:21
Station 2 - 00:08:44 - 00:13:24
Station 3 - 00:00:00 - 00:00:00
acc - 64.67%
balanced acc - 77.43%
-------------------------------------

Devamma 1931981 EUS.avi (Old Olympus)
src - ./testing/data/Devamma 1931981 EUS.avi
crop_params = [185, 105, 93, 90] 
trained - no
test - yes
Station 1 - 00:15:33 - 00:19:12
Station 2 - 00:00:00 - 00:08:40
Station 3 - 00:09:07 - 00:10:21
Station 3 - 00:12:26 - 00:13:00
acc - 64.05%
balanced acc - 64.88%
-------------------------------------

video_mammen.mkv
crop_params = [140, 420, 105, 103]
trained -
test -
Station 1 - 00:00:00 - 00:00:00
Station 2 - 00:00:00 - 00:00:00
Station 3 - 00:00:00 - 00:00:00
acc - 0.00%
balanced acc - 0.00%
'''

# Config
LIVE_INFERENCE = False
cv_src = 2 if LIVE_INFERENCE else Vid_source  # 2 for HDMI port
IMAGE_SIZE = 224
crop_params = [185, 120, 100, 90] # left, right, top, bottom
end_time = 1 + 60 * 15 # 20 minutes

ground_truth = [
    {'class':'1', 'start': '00:15:33', 'end': '00:19:12'},
    {'class':'2', 'start': '00:00:00', 'end': '00:08:40'},
    {'class':'3', 'start': '00:09:07', 'end': '00:10:21'},
    {'class':'3', 'start': '00:12:26', 'end': '00:13:00'}
]


cap = cv2.VideoCapture(cv_src, cv2.CAP_FFMPEG)

log.info("Video capture object initialized.")

capture_fps = int(cap.get(cv2.CAP_PROP_FPS))
log.info(f"Capture FPS: {capture_fps}")

width, height = int(cap.get(3)), int(cap.get(4))

if not cap.isOpened():
    log.error("Could not access the video source.")
    exit()


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

def avg_pixel(image):
    average = image.mean()
    return average


def red_pixel_avg(image, hsv):
    lower_red = np.array([-10, 100, 100])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(image, image, mask=mask)

    average = np.mean(res)

    return average


def orange_pixel_avg(image, hsv):

    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    res = cv2.bitwise_and(image, image, mask=mask)

    average = np.mean(res)

    return average


def blue_pixel_sum(image, hsv):

    lower_blue = np.array([100, 150, 0], np.uint8)
    upper_blue = np.array([140, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(image, image, mask=mask)

    average = np.mean(res)

    return average

def avg_green(image, hsv):

    lower_green = np.array([36, 25, 25], np.uint8)
    upper_green = np.array([70, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.bitwise_and(image, image, mask=mask)

    average = np.mean(green_pixels)

    return average

def isDisturbed(frame):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if (red_pixel_avg(frame, hsv_image) >= 0.00068 or blue_pixel_sum(frame, hsv_image) >= 0.0094):
        return True, "Doppler"
    
    # if (red_pixel_avg(frame, hsv_image) >= 20 or orange_pixel_avg(frame, hsv_image) >= 20):
    #     return True, "Camera feed"
    
    if (avg_pixel(frame) < 12):
        return True, "Black frame"
    
    if (avg_green(frame, hsv_image) > 0.21):
        return True, "Green Cursor"
        
    return False, 'None'


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

station_class = {0: '1', 1: '2', 2: '3'}

correct_predictions = 0
total_predictions = 0
actual = []
predicted = []

with torch.no_grad():
        start_time = time.time()
        init_time = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                log.error(" Unable to capture frame.")
                break

            left, right, top, bottom = crop_params
            frame = frame[top:height - bottom, left:width - right]


            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) 
            current_time = timedelta(milliseconds=current_time)
            current_time_format = f"{current_time.seconds//3600:02}:{(current_time.seconds%3600)//60:02}:{current_time.seconds%60:02}"

            
            if current_time.seconds > end_time:
                log.info("End of video reached.")
                log.info(f"Accuracy: {correct_predictions/total_predictions*100:.2f}%")
                break

            cv2.putText(
                    frame,
                    current_time_format,
                    (width - 387, height - 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            disturbance, reason = isDisturbed(frame)

            if not disturbance:
                frame_tensor = preprocess_frame(frame)

                prediction = model(frame_tensor.unsqueeze(0)).squeeze(0).softmax(0)
                prediction_class = prediction.argmax().item()
                log.info(f': Station {station_class[prediction_class]} - at {current_time_format}')

                for entry in ground_truth:
                    if entry['start'] <= current_time.total_seconds() <= entry['end']:
                        if station_class[prediction_class] == entry['class']:
                            log.info(f"Correct prediction: {station_class[prediction_class]}")
                            correct_predictions += 1
                        else:
                            log.info(f"Incorrect prediction: {station_class[prediction_class]}, actual: {entry['class']}")
                        
                        actual.append(entry['class'])
                        predicted.append(station_class[prediction_class])
                        total_predictions += 1

                for i in range(3):
                    text = f"Station {station_class[i]}: {prediction[i].item()*100:.2f}%"
                    position = (10, (i + 1) * 30)
                    color = (255, 255, 255)
                    cv2.putText(
                        frame,
                        text,
                        position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
            else:
                log.info(f"Disturbed frame. Skipping frame. Reason: {reason}")
        

            cv2.imshow("EUS Inference", frame)

            elapsed_time = time.time() - start_time

            init_time += elapsed_time

            cap.set(cv2.CAP_PROP_POS_MSEC, (init_time) * 1000 )
            log.debug(f'moving to {elapsed_time}')
            start_time = start_time + elapsed_time

            if cv2.waitKey(capture_fps) & 0xFF == ord("q"):
                log.info("Exiting.")
                break

cap.release()
cv2.destroyAllWindows()

balanced_accuracy = balanced_accuracy_score(actual, predicted)    
log.info(f"Balanced accuracy: {balanced_accuracy*100:.2f}%")
