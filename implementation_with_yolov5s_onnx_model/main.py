import sys
sys.path.insert(1,"/home/nerd/Desktop/AI-based-Traffic-Control-System--/common")

import cv2
import utils as util
import time
import numpy as np

vs  = cv2.VideoCapture("/home/nerd/Desktop/AI-based-Traffic-Control-System--/datas/video1.mp4")
vs2 = cv2.VideoCapture("/home/nerd/Desktop/AI-based-Traffic-Control-System--/datas/video1.mp4")
vs3 = cv2.VideoCapture("/home/nerd/Desktop/AI-based-Traffic-Control-System--/datas/video1.mp4")
vs4 = cv2.VideoCapture("/home/nerd/Desktop/AI-based-Traffic-Control-System--/datas/video1.mp4")


net = cv2.dnn.readNet("/home/nerd/Desktop/AI-based-Traffic-Control-System--/models/yolov5s.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
ln = net.getUnconnectedOutLayersNames()



while True:

        # read the next frame from the 
        (success, frame) = vs.read()
        (success, frame2) = vs2.read()
        (success, frame3) = vs3.read()
        (success, frame4) = vs4.read()
        # if the frame was not successfuly captured, then we have reached the end
        # of the stream or there is disconnection
        if not success:
                break
        start = time.time()
        counts,lanes = util.final_output(net,ln,[frame,frame2,frame3,frame4])
        end = time.time()
        print("total processing:"+str(end-start))
        cv2.imshow("f",lanes[0])
        cv2.waitKey(1)
        
                



