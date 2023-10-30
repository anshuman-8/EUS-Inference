import cv2
import numpy as np
from PIL import Image
import logging as log
from torchvision import transforms

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

    if (red_pixel_avg(frame, hsv_image) >= 3 or orange_pixel_avg(frame, hsv_image) >= 5 or avg_green(frame, hsv_image) >= 50):
        return True, "Camera feed"
    
    if (red_pixel_avg(frame, hsv_image) >= 3 or blue_pixel_sum(frame, hsv_image) >= 0.094):
        return True, "Doppler"
    
    if (avg_pixel(frame) < 9):
        return True, "Black frame"
    
    if (avg_green(frame, hsv_image) > 0.31):
        return True, "Green Cursor"
        
    return False, 'None'


transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
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
