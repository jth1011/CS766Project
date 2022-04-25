import cv2
import os
import imutils
import random
import numpy as np


class videoLoader:

    def __init__(self, path):
        self.img_path = os.path.join(path, "img")
        self.imgs = [img for img in os.listdir(self.img_path) if img.endswith(".jpg")]
        self.truth = np.loadtxt(os.path.join(path, "groundtruth.txt"), delimiter=',')
        frame = cv2.imread(os.path.join(self.img_path, self.imgs[0]))
        height, width, _ = frame.shape
        self.size = (width, height)
        assert len(self.imgs) == self.truth.shape[0]

    def get_imgs(self):
        return self.imgs

    def get_truth(self):
        return self.truth

    def get_size(self):
        return self.size


class videoSplitter:

    def __init__(self, rot, crop, height):
        self.rot1 = random.randint(-rot, rot)
        self.rot2 = random.randint(-rot, rot)
        self.crop = crop
        self.height = height

    def crop_rot(self, frame):
        frame_left = imutils.rotate(frame, self.rot1)
        frame_right = imutils.rotate(frame, self.rot2)
        frame_left = frame_left[:self.height, :self.crop]
        frame_right = frame_right[:self.height, -self.crop:]
        return frame_left, frame_right
