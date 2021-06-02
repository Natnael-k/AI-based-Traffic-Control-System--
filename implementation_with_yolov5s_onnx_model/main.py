import sys
import argparse
import pathlib
sys.path.insert(1,str(pathlib.Path.cwd().parents[0])+"/common")

import cv2
import utils as util
import time
import numpy as np


def main(sources):
	#read image from each lanes video source
	vs  = cv2.VideoCapture(str(pathlib.Path.cwd().parents[0])+"/datas/"+sources[0])
	vs2 = cv2.VideoCapture(str(pathlib.Path.cwd().parents[0])+"/datas/"+sources[1])
	vs3 = cv2.VideoCapture(str(pathlib.Path.cwd().parents[0])+"/datas/"+sources[2])
	vs4 = cv2.VideoCapture(str(pathlib.Path.cwd().parents[0])+"/datas/"+sources[3])

	#creates a network given yolov5s model
	net = cv2.dnn.readNet(str(pathlib.Path.cwd().parents[0])+"/models/yolov5s.onnx")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	ln = net.getUnconnectedOutLayersNames() # returns the name of output layer

	#initial configuration of each lanes order
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
		# assigns each lane its corresponding frame
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
		lanes = util.final_output(net,ln,lanes) # returns lanes object with processed frame
		end = time.time()
		print("total processing:"+str(end-start))
		if wait_time<=0:
		   images_transition=util.display_result(wait_time,lanes)    
		   final_image = cv2.resize(images_transition,(1020,720))
		   cv2.imshow("f",final_image)
		   cv2.waitKey(100)
		   
		    
		   wait_time=util.schedule(lanes) # returns waiting duration of each lane
		images_scheduled=util.display_result(wait_time,lanes)    
		final_image = cv2.resize(images_scheduled,(1020,720))
		cv2.imshow("f",final_image)
		cv2.waitKey(1)
		wait_time=wait_time-1
if __name__=="__main__":
	
        
        
        parser = argparse.ArgumentParser(description="Determines duaration based on car count on images")
        parser.add_argument("--sources",help="video feeds to be infered on, the videos must reside in the datas folder", type=String,default="video1.mp4,video5.mp4,video2.mp4,video3.mp4") 
        args = parser.parse_args()

        sources=args.sources
        sources =sources.split(",")
        print(type(sources))
        main(sources)		        



