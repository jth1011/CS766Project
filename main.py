import cv2
import os
import stitch
import video

if __name__ == "__main__":
    ratio_param = 0.5
    thresh_param = 4.0
    iter_param = 250
    data_path_list = [os.path.join("data","airplane-"+str(i)) for i in range(1,21)]
    for data in data_path_list:
        vl = video.videoLoader(data)
        (width, height) = vl.get_size()
        imgs = vl.get_imgs()
        vs = video.videoSplitter(30, int(width*0.6), height)
        pano = stitch.stitcher(ratio=ratio_param, ransacThresh=thresh_param, maxIters=iter_param)
        for i, img in enumerate(imgs):
            img_path = os.path.join(data,"img",img)
            (left, right) = vs.crop_rot(cv2.imread(img_path))
            if i == 0:
                h = pano.match(left, right)
            result = pano.combine(left, right, h, blend=False)
            cv2.imshow("result",result)
            cv2.waitKey(1)
