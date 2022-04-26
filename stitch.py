import cv2
import numpy as np

class stitcher:

    def __init__(self, matches=16, ratio=0.8, ransacThresh=5.0, maxIters=1000):
        self.min_matches = matches
        self.ratio = ratio
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
            h_3x3, status = cv2.findHomography(img2_kp, img1_kp, cv2.RANSAC,
                        ransacReprojThreshold=self.ransacThresh, maxIters=self.ransacIters)
        else:
            print("Not enough optimal points found")
        return h_3x3

    def combine(self, img0, img1, h_matrix, blend):

        points0 = np.float32([[0, 0], [0, img0.shape[0]], [img0.shape[1],
                        img0.shape[0]], [img0.shape[1], 0]]).reshape((-1, 1, 2))

        points1 = np.float32([[0, 0], [0, img1.shape[0]], [img1.shape[1],
                        img1.shape[0]], [img1.shape[1], 0]],).reshape((-1, 1, 2))

        points2 = cv2.perspectiveTransform(points1, h_matrix)
        points = np.concatenate((points0, points2), axis=0)

        [x_min, y_min] = (points.min(axis=0).ravel() - 0.5).astype(np.int32)
        [x_max, y_max] = (points.max(axis=0).ravel() + 0.5).astype(np.int32)

        h_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        output_img = cv2.warpPerspective(img1, h_translation.dot(h_matrix),
                                (x_max - x_min, y_max - y_min), flags=cv2.INTER_LINEAR)

        if blend:
            output_img2 = np.zeros_like(output_img)
            output_img2[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = img0
            return self.mean_blend_smooth(output_img, output_img2)
        else:
            output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = img0
            return output_img

    def mean_blend(self, img1, img2):
        assert (img1.shape == img2.shape)
        locs1 = np.where(img1 != 0)
        blended1 = np.copy(img2)
        blended1[locs1[0], locs1[1]] = img1[locs1[0], locs1[1]]
        locs2 = np.where(img2 != 0)
        blended2 = np.copy(img1)
        blended2[locs2[0], locs2[1]] = img2[locs2[0], locs2[1]]
        blended = cv2.addWeighted(blended1, 0.5, blended2, 0.5, 0)
        return blended

    def mean_blend_smooth(self, img1, img2):
        assert (img1.shape == img2.shape)

        # Create distance map for image 1
        locs1 = np.where(img1 != 0)
        mask1 = np.zeros(img1.shape[:2], dtype="uint8")
        mask1[locs1[0], locs1[1]] = 255
        # These closing operations don't seem to make much of a difference...
        #mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        d1 = cv2.distanceTransform(mask1, distanceType=cv2.DIST_C, maskSize=cv2.DIST_MASK_3)
        d1 = d1 / np.max(d1)

        # Create distance map for image 2
        locs2 = np.where(img2 != 0)
        mask2 = np.zeros(img2.shape[:2], dtype="uint8")
        mask2[locs2[0], locs2[1]] = 255
        #mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        d2 = cv2.distanceTransform(mask2, distanceType=cv2.DIST_C, maskSize=cv2.DIST_MASK_3)
        d2 = d2 / np.max(d2)

        # Compute the blending weights for each image using distance maps
        sum = d1 + d2
        weight1 = np.nan_to_num(d1 / sum)
        weight2 = np.nan_to_num(d2 / sum)
        print(img1.shape," ",img2.shape)
        print(weight1.shape," ",weight2.shape)
        print(np.multiply(img1, weight1).shape)
        print(np.multiply(img2, weight2).shape)

        # Blend the two images together according to their weights
        blended = np.rint(np.multiply(img1, weight1) + np.multiply(img2, weight2))
        blended = blended.astype(np.uint8)
        return blended

    def crop(self, img):
        height, width = img.shape[:2]
        lim = round(height*0.05)
        return img[lim:-lim,:]

    def getSIFTFeatures(self, img):
        kp, des = self.sift.detectAndCompute(img, None)
        return {'kp':kp, 'des':des}
