from PIL import Image
import cv2
import os
import numpy as np

kelvin_table = {
    1000: (255,56,0),
    1500: (255,109,0),
    2000: (255,137,18),
    2500: (255,161,72),
    3000: (255,180,107),
    3500: (255,196,137),
    4000: (255,209,163),
    4500: (255,219,186),
    5000: (255,228,206),
    5500: (255,236,224),
    6000: (255,243,239),
    6500: (255,249,253),
    7000: (245,243,255),
    7500: (235,238,255),
    8000: (227,233,255),
    8500: (220,229,255),
    9000: (214,225,255),
    9500: (208,222,255),
    10000: (204,219,255)}


def convert_temp(image, temp):
    r, g, b = kelvin_table[temp]

    image[:,:,0] = np.multiply(image[:,:,0], b / 255.0)
    image[:,:,1] = np.multiply(image[:,:,1], g / 255.0)
    image[:,:,2] = np.multiply(image[:,:,2], r / 255.0)

    image = np.round(image)

    return image.astype(np.uint8)


def apply_vignette(image, effect_intensity=1):
    vig = cv2.imread(os.path.join("resources", "vignette_mask.png"))
    x = image.shape[0]
    y = image.shape[1]
    new_img = np.round(np.multiply(image, np.multiply(np.divide(cv2.resize(vig, (y, x)), 255), effect_intensity) + (1 - effect_intensity)))
    new_img = new_img.astype(np.uint8)

    return new_img