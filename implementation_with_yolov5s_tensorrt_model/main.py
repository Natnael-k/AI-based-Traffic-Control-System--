import sys
sys.path.insert(1,"/home/nerd/Desktop/AI-based-Traffic-Control-System--/common")

import cv2
import utils as util
import time
import numpy as np
from Processor import Processor

vs  = cv2.VideoCapture("/home/nerd/Desktop/AI-based-Traffic-Control-System--/datas/video1.mp4")
vs2 = cv2.VideoCapture("/home/nerd/Desktop/AI-based-Traffic-Control-System--/datas/video1.mp4")
vs3 = cv2.VideoCapture("/home/nerd/Desktop/AI-based-Traffic-Control-System--/datas/video1.mp4")
vs4 = cv2.VideoCapture("/home/nerd/Desktop/AI-based-Traffic-Control-System--/datas/video1.mp4")


processor = Processor("yolov5s.trt")

lanes = util.Lanes([util.Lane("","",1),util.Lane("","",3),util.Lane("","",4),util.Lane("","",2),])
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
        lanes = util.final_output_tensorrt(processor,lanes)
        end = time.time()
        print("total processing:"+str(end-start))
        if wait_time<=0:
           images_transition=util.display_result(wait_time,lanes)    
           final_image = cv2.resize(images_transition,(1020,720))
           cv2.imshow("f",final_image)
           cv2.waitKey(10)
           
            
           wait_time=util.schedule(lanes)
        images_scheduled=util.display_result(wait_time,lanes)    
        final_image = cv2.resize(images_scheduled,(1020,720))
        cv2.imshow("f",final_image)
        cv2.waitKey(1)
        wait_time=wait_time-1
                



