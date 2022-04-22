import cv2

class tracker:

    def __init__(self):
        self.tracker = cv2.TrackerKCF_create()
        return