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

lanes = util.Lanes([util.Lane("","",2),util.Lane("","",3),util.Lane("","",4),util.Lane("","",1),])
wait_time=0

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
        for i,lane in enumerate(lanes.getLanes()):
            if(lane.lane_number==1):
                lane.frame=frame
            elif(lane.lane_number==2):
                lane.frame=frame2
            elif(lane.lane_number==3): 
                lane.frame=frame3
            elif(lane.lane_number==4):
                lane.frame= frame4
        start = time.time()
        if wait_time<=1:
            
            wait_time,frame= util.final_output(net,ln,lanes)
        end = time.time()
        print("total processing:"+str(end-start))
        frame = cv2.putText(frame,"Green:"+str(wait_time),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("f",frame)
        cv2.waitKey(1)
        wait_time=wait_time-1
                



