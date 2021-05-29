
import numpy as np
import cv2
import time



"""
a blueprint for a bounded box with its corresponding name,confidence score and 
"""


class BoundedBox:
    
    def __init__(self, xmin, ymin, xmax, ymax, ids, confidence):
        with open("/home/nerd/Desktop/AI-based-Traffic-Control-System--/datas/coco.names", 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')  # stores a list of classes
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax       
        self.ymax = ymax
        self.name = self.classes[ids]
        self.confidence = confidence   


"""
a blueprint that has lanes as lists and give queue like functionality 
to reorder lanes based on their turn for green and red light state
"""

class Lanes:
    def __init__(self,lanes):
        self.lanes=lanes
    
    def getLanes(self):
        
        return self.lanes
    
    def lanesTurn(self):
        
       return self.lanes.pop(0)

    def enque(self,lane):
 
       return self.lanes.append(lane)
    def lastLane(self):
       return self.lanes[len(self.lanes)-1]
 
class Lane:
    def __init__(self,count,frame,lane_number):
        self.count = count
        self.frame = frame
        self.lane_number = lane_number
    
 
def schedule(lanes):
   
    standard=10
    reward =0
    turn = lanes.lanesTurn()
    
    for i,lane in enumerate(lanes.getLanes()):
        if(i==(len(lanes.getLanes())-1)):
            reward = reward + (turn.count-lane.count)*0.2
        else:
            reward = reward + (turn.count-lane.count)*0.5
    scheduled_time = standard+reward
    lanes.enque(turn)
    return scheduled_time
       
    

def display_result(wait_time,lanes):
    green = (0,255,0)
    red  = (0,0,255)
    yellow= (0,255,255)
    for i ,lane in enumerate(lanes.getLanes()):
        if(wait_time<=0 and (i==(len(lanes.getLanes())-1) or i==0)):
           color=yellow
           text="yellow:2 sec"
           
        elif(wait_time>=0 and i==(len(lanes.getLanes())-1)):
            color = green 
            text="green:"+str(wait_time)+" sec"
            print(text)
        else:
            color=red
            text="red:"+str(wait_time)+ " sec"
        lane.frame = cv2.putText(lane.frame,text,(60,105),cv2.FONT_HERSHEY_SIMPLEX,4,color,6)
        lane.frame = cv2.putText(lane.frame,"car count:"+str(lane.count),(60,195),cv2.FONT_HERSHEY_SIMPLEX,3,color,5)
        globals()['img%s' % lane.lane_number]=lane.frame
    hori_image = np.concatenate((img1, img2), axis=1)
    hori2_image = np.concatenate((img3, img4), axis=1)
    all_lanes_image = np.concatenate((hori_image, hori2_image), axis=0)
    
    return all_lanes_image




def vehicle_count(Boxes):
        vehicle=0
        for box in Boxes:
            if box.name == "car" or box.name == "truck" or box.name == "bus":
                vehicle=vehicle+1  

        return vehicle


def _make_grid(nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

def drawPred( frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), thickness=6)

        return frame

def postprocess(frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / 320, frameWidth / 320
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > 0.5 and detection[4] > 0.5:
                    center_x = int(detection[0] * ratiow)
                    center_y = int(detection[1] * ratioh)
                    width = int(detection[2] * ratiow)
                    height = int(detection[3] * ratioh)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
        correct_boxes = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            box=BoundedBox(box[0],box[1],box[2],box[3],classIds[i],confidences[i])
            correct_boxes.append(box)
            frame = drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return correct_boxes,frame


"""
interpret the ouptut boxes into the appropriate bounding boxes based on the yolo paper 
logspace transform
"""
def modify(outs,confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        with open('/home/nerd/Desktop/AI-based-Traffic-Control-System--/datas/coco.names', 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')   
        colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(classes))]
        num_classes = len(classes)
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        nl = len(anchors)
        na = len(anchors[0]) // 2
        no = num_classes + 5
        grid = [np.zeros(1)] * nl
        stride = np.array([8., 16., 32.])
        anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, 1, -1, 1, 1, 2)

        
        z = []  # inference output
        for i in range(nl):
            bs, _, ny, nx,c = outs[i].shape  
            if grid[i].shape[2:4] != outs[i].shape[2:4]:
                grid[i] = _make_grid(nx, ny)
                

            y = 1 / (1 + np.exp(-outs[i])) 
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * int(stride[i])
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1,no))
        z = np.concatenate(z, axis=1)
        return z

def final_output(net,output_layer,lanes):
        
        for lane in lanes.getLanes():
            blob = cv2.dnn.blobFromImage(lane.frame, 1 / 255.0, (320, 320),
                swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(output_layer)
            end = time.time() 
            print("fps:"+str(end-start))   
            dets = modify(layerOutputs)
            start = time.time()
            boxes,frame = postprocess(lane.frame,dets)
            end = time.time()
            print("post_process:"+str(end-start))
            start = time.time()
            count = vehicle_count(boxes)
            lane.count= count
            print(lane.count)
            lane.frame=frame
            end = time.time()
            print("counting and drawing:"+str(end-start))
        
        
        return lanes
