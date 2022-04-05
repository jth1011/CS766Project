import cv2
import numpy as np
import argparse

import stitch
import video

if __name__ == "__main__":
    pano = stitch.stitcher(ratio=0.75, ransacThresh=4.0, maxIters=1000)
    im1 = cv2.imread("imgs/jackson_image1_lr.jpg", cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread("imgs/jackson_image2_lr.jpg", cv2.IMREAD_GRAYSCALE)
    im3 = cv2.imread("imgs/jackson_image3_lr.jpg", cv2.IMREAD_GRAYSCALE)
    h12 = pano.match(im2, im1)
    result = pano.combine(im2, im1, h12, blend=True)
    h23 = pano.match(result, im3)
    result = pano.combine(result, im3, h23, blend=True)
    cv2.imshow("result", pano.crop(result))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #cam1 = video.camThread("Camera 1", 0)
    #cam2 = video.camThread("Camera 2", 1)

    #cam1.start()
    #cam2.start()