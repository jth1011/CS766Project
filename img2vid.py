import cv2
import os

end = 10

for i in range (1,end+1):
    path = "airplane-"+str(i)+"/img"
    vidname = "video"+str(i)+".mp4"
    
    imgs = [img for img in os.listdir(path) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(path, imgs[0]))
    height, width, layers = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    writer = cv2.VideoWriter(vidname, fourcc=fourcc, fps=30, frameSize=(width,height))
    
    print("Start to write " + vidname)
    for image in imgs:
        writer.write(cv2.imread(os.path.join(path, image)))
    
    print("Finished writing " + vidname)
    writer.release()