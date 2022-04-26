import cv2
import os
import imutils
import random
import numpy as np


class videoLoader:

    def __init__(self, path):
        self.img_path = os.path.join(path, "img")
        self.imgs = [img for img in os.listdir(self.img_path) if img.endswith(".jpg")]
        self.truth = np.loadtxt(os.path.join(path, "groundtruth.txt"), delimiter=',', dtype=np.int16)
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

    def __init__(self, rot, trans, crop, bright, height, width):
        self.rot = random.randint(-rot, rot)
        self.trans_x = random.randint(-trans, trans)
        self.trans_y = random.randint(-trans, trans)
        self.bright = random.randint(bright[0], bright[1])
        self.crop = crop
        self.height = height

    def crop_rot(self, frame):
        # left frame is set as base image
        frame_left = frame

        # add brightness constant to right frame
        frame_right = cv2.convertScaleAbs(frame, alpha=1, beta=self.bright)

        # translate the right frame by random x&y values
        frame_right = imutils.translate(frame_right, self.trans_x, self.trans_y)

        # change the rotation of the image by a small random value and cap if too large or small
        self.rot = self.rot + random.randint(-2,2)
        self.rot = max(min(self.rot, 30), -30)
        frame_right = imutils.rotate(frame_right, self.rot)

        #crop the left and right frames to make the amount of overlap smaller
        frame_left = frame_left[:, :self.crop]
        frame_right = frame_right[:, -self.crop:]
        return frame_left, frame_right, frame
