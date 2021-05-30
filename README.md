# AI-based-Traffic-Light-Control-System
Computer vision aided traffic light scheduling systems



AI-based-Traffic-Light-Control-System is an inteligent embedded system which applies computer vision to determine the density of cars at each lane on a traffic intersection so as to generate adaptive duration for green and red traffic light at each lane. 

This repository represents an ongoing open source research into utilizing different object detection algorithims like YOLO  to design an inteligent and adaptive  traffic light control system. All the code and models are under research and development and subject to change.




Yolov5s is selected for this project due to its speed, lightness and accuracy. The yolov5s model can be found from https://github.com/ultralytics/yolov5 

While the models speed is great, it is not efficent enough to be deployed on edge devices for inference. To take advantage of performance the model is exported into onnx version and then exported to Tensorrt model which optimizes the model for inference. The performance of the model before and after optimization is shown below. Tutorials on how to export Yolov5s model into tensorrt model can be found on the tutorial section at https://github.com/ultralytics/yolov5




This  comparison is tested on jetson nano

| Detection Algorithim     | Platform | FPS    |
| :---        |    :----:   |          ---: |
| Yolov5s      | Pytorch       | 0.32   |
| Yolov5s    | ONNX        | 0.25      |
| Yolov5s    | Tensorrt        | 0.08    |
| Yolov4      | Darknet       | -   |
| Yolov3    | Darknet        | -     |
| Yolov3-tiny    | Darknet        |  -      |



## Devices Used

- Nvidia Jetson Nano
- Ip camera


## Features

- Detect and counts vehicles from a camera feed on each lane
- Determine a green and red light duration based on comparison of each lanes vehicle density
- Displays a simulation


## Work flow




## Project directory
```
project
│   README.md
│   requirement.txt    
│
|__ common
│   │  utils.py
|__ datas
    │   video1.py
    │   coco.name
|__ implementation_with_yolov5s_onnx_model
    |   main.py
|__ implementation_with_yolov5s_tensorrt_model
    |  processor.py
    |  main.py
|__ models
    | yolov5s.onnx
    | yolov5s.trt
```


## Getting started
