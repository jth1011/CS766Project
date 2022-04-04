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

    def combine(self, img1, img2, h_3x3):
        w1, h1 = img1.shape[:2]
        w2, h2 = img2.shape[:2]
        pts1 = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)
        new_pts2 = cv2.perspectiveTransform(pts2, h_3x3)
        pts = np.concatenate((pts1, new_pts2), axis=0)

        [x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5)
        transform = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        result = cv2.warpPerspective(img2, transform.dot(h_3x3),
                                     (x_max-x_min, y_max-y_min))

        result[-y_min:-y_min + w1, -x_min:-x_min + h1] = img2
        return result

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
    result = stitcher.combine(img1=im1, img2=im2, h_3x3=h)
    cv2.imshow("result", result)
    cv2.waitKey(0)
