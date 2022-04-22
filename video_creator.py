import cv2
import random
from scipy import ndimage

def crop_video(vidCap, vidWritL, vidWritR, height, crop_val, rot):
    crop1 = random.randint(-rot, rot)
    crop2 = random.randint(-rot, rot)
    
    while(True):
        ret, frame = vidCap.read()
        
        if not ret:
            break
        
        frame_left = ndimage.rotate(frame, crop1)
        frame_right = ndimage.rotate(frame, crop2)
        frame_left = frame_left[:height,:crop_val]
        frame_right = frame_right[:height,-crop_val:]
        vidWritL.write(frame_left)
        vidWritR.write(frame_right)
        
    

def main():
    filename = "pexels.mp4"
    writepathL = "cropped_pexels_left.mp4"
    writepathR = "cropped_pexels_right.mp4"
    cap = cv2.VideoCapture(filename)
    width  = cap.get(3)
    height = cap.get(4)
    crop_val = round(width*0.75)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    writ_left = cv2.VideoWriter(filename=writepathL, fourcc=fourcc, fps=30, frameSize=(crop_val, int(height)))
    writ_right = cv2.VideoWriter(filename=writepathR, fourcc=fourcc, fps=30, frameSize=(crop_val, int(height)))
    crop_video(cap, writ_left, writ_right, int(height), crop_val, 6)
    writ_left.release()
    writ_right.release()
    cap.release()
    
if __name__ == '__main__':
    main()