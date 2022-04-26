import cv2
import os
import stitch
import video
import track
import imageio
import numpy as np

if __name__ == "__main__":
    ratio_param = 0.5
    thresh_param = 4.0
    iter_param = 250
    data_path_list = [os.path.join("data","airplane-"+str(i)) for i in range(1,21)]
    for data in data_path_list:

        vl = video.videoLoader(data)
        (width, height) = vl.get_size()
        imgs = vl.get_imgs()
        truth = vl.get_truth()
        vs = video.videoSplitter(rot=90, trans=10, crop=int(width*0.6), bright=(10,20), height=height, width=width)
        tr = track.tracker(option=3)
        pano = stitch.stitcher(ratio=ratio_param, ransacThresh=thresh_param, maxIters=iter_param)

        ious = np.zeros(99)
        for i, img in enumerate(imgs):
            if i == 100:
                break
            img_path = os.path.join(data,"img",img)
            (left, right, orig) = vs.crop_rot(cv2.imread(img_path))
            h = pano.match(left, right)
            result = pano.combine(left, right, h, blend=False)

            # crops image to remove black borders
            # needed to make sure original bounding boxes are correct
            y_nz, x_nz, _ = np.nonzero(result)
            result = result[np.min(y_nz):np.max(y_nz), np.min(x_nz):np.max(x_nz), :]

            if i == 0:
                tr.initialize(img=result, box=tuple(truth[i]))
            else:
                ok, box = tr.update(result)
                box = tuple(map(int, box))
                iou = tr.get_iou(box, truth[i, :])
                ious[i-1] = iou
                cv2.putText(result, "IOU: "+str(iou), (50,50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),2,1)
                cv2.rectangle(result, truth[i,:2], np.add(truth[i,:2], truth[i,2:]), (255, 0, 0), 2, 1)
                cv2.rectangle(result, box[:2], np.add(box[:2], box[2:]), (0, 255, 0), 2, 1)
                cv2.imshow("result",result)
                cv2.waitKey(1)
        print(ious.mean())
        break