import cv2
import numpy as np

class stitch:

    def __init__(self, matches=10, ratio=0.75, window_size=500, ransacThresh=1.0, maxIters=1000):
        self.min_matches = matches
        self.ratio = ratio
        self.smoothing_window = window_size
        self.ransacThresh = ransacThresh
        self.ransacIters = maxIters
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher()

    def match(self, img1, img2):
        feat1 = self.getSIFTFeatures(img1)
        feat2 = self.getSIFTFeatures(img2)
        matches = self.matcher.knnMatch(feat1['des'], feat2['des'], k=2)
        optimal_points = [(p1.trainIdx, p1.queryIdx) for p1,p2 in matches
                          if p1.distance <= self.ratio * p2.distance]
        if len(optimal_points) >= self.min_matches:
            img1_kp = np.float32(
                [feat1['kp'][i].pt for (_, i) in optimal_points] )
            img2_kp = np.float32(
                [feat2['kp'][i].pt for (i, _) in optimal_points] )
            h_3x3, status = cv2.findHomography(img1_kp, img2_kp, cv2.RANSAC,
                        ransacReprojThreshold=self.ransacThresh, maxIters=self.ransacIters)
        else:
            print("Not enough optimal points found")
        return h_3x3

    def mask(self, img1, img2):
        return 0

    def blend(self, img1, img2):
        return 0

    def getSIFTFeatures(self, img):
        kp, des = self.sift.detectAndCompute(img, None)
        return {'kp':kp, 'des':des}

if __name__ == "__main__":
    stitcher = stitch()
    im1 = cv2.imread("mountain_left.png", cv2.IMREAD_COLOR)
    im2 = cv2.imread("mountain_center.png", cv2.IMREAD_COLOR)
    img1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    h = stitcher.match(img1_gray, img2_gray)
    print(h)