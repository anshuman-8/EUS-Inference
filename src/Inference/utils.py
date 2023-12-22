import cv2
import numpy as np
from PIL import Image
import logging as log
from torchvision import transforms


def avg_pixel(image):
    average = image.mean()
    return average


def red_pixel_sum(image, hsv):
    lower_red = np.array([-10, 100, 100])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(image, image, mask=mask)

    sum = np.sum(res)

    return sum

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

    sum = np.sum(res)

    return sum


def green_pixel_sum(image, hsv):
    lower_green = np.array([36, 25, 25], np.uint8)
    upper_green = np.array([70, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.bitwise_and(image, image, mask=mask)

    sum = np.sum(green_pixels)

    return sum

def avg_green(image, hsv):
    lower_green = np.array([36, 25, 25], np.uint8)
    upper_green = np.array([70, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.bitwise_and(image, image, mask=mask)

    average = np.mean(green_pixels)

    return average


def isDisturbed(frame):
    """
    Checks if the image is disturbed, i.e. if image is Camera feed, Doppler, Black frame or Green Cursor
    """
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if (
        red_pixel_avg(frame, hsv_image) >= 20
        or orange_pixel_avg(frame, hsv_image) >= 20
    ):
        return True, "Camera feed"

    if (
        blue_pixel_sum(frame, hsv_image) >= 100000
        or green_pixel_sum(frame, hsv_image) >= 50000
    ):
        return True, "Doppler"

    if avg_pixel(frame) < 9:
        return True, "Black frame"

    if avg_green(frame, hsv_image) > 0.31:
        return True, "Green Cursor"

    return False, "None"


def preprocess_frame(frame, image_size, mean, std):
    """Preprocess the frame"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    return transform(frame)


def crop_frame(frame, crop_dim):
    """Crop the frame to the given dimensions [T, B, L, R]"""
    frame = frame[crop_dim[0] : -(crop_dim[1] + 1), crop_dim[2] : -(crop_dim[3] + 1)]
    return frame
