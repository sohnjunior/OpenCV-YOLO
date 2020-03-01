OpenCV YOLO
============ 

<br>

Installation
----------------
#### Required packages
```
pip install -r requirements.txt
```
> opencv-python-headless 는 pycharm 환경에서 필요한 패키지입니다.

#### YOLO
Download pre-trained weight file(yolov3.weights)
```
$ wget https://pjreddie.com/media/files/yolov3.weights
```
Download yolo3 configuration file(yolov3.cfg), and object class file(yolov3.txt) at below <br>
https://pjreddie.com/darknet/yolo/

<br>

Usage
--------------- 
```
python opencv_yolo.py
```