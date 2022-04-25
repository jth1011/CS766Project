import cv2


class tracker:

    def __init__(self):
        self.tracker = cv2.TrackerMIL_create()

    def initialize(self, img, box):
        self.tracker.init(img, box)

    def update(self, frame):
        return self.tracker.update(frame)

    def get_iou(self, box, truth):
        xA = max(box[0], truth[0])
        yA = max(box[1], truth[1])
        xB = min(box[0]+box[2], truth[0]+truth[2])
        yB = min(box[1]+box[3], truth[1]+truth[3])

        interArea = (xB - xA) * (yB - yA)

        boxAArea = box[2] * box[3]
        boxBArea = truth[2] * truth[3]

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou