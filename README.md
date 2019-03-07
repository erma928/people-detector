# People Detector

## Introduction
The program detects people in more or less still motion in video surveillance. 
It can use the common appoarch that makes uses of mog background modeling and subtraction process for contour detection. 
It can also utilizes DNN approaches like Darknet YOLO or Mobilenet SSD for more accurate detections. The contour detection 
has been run in ARM environment as well.

## Program Build

### Dependencies
* `cmake`
* `opencv >= 3.4.5`
* [ARM Linux OpenCV enviroment settings](https://docs.opencv.org/master/d0/d76/tutorial_arm_crosscompile_with_cmake.html)

### Build
Extract to `people-detector` directory

    cd people-detector/
    mkdir build
    cmake ..
    make 

### Program Run

    fengjimin$ ./people-detector 
    people detection using deep learning networks or common contour detection using OpenCV.
    Usage: people-detector [params] alias 
    
    	--backend (value:0)
    		Choose one of computation backends: 0: automatically (by default), 1: Halide language (http://halide-lang.org/), 2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), 3: OpenCV implementation
    	-c, --config
    		Path to a text file of model contains network configuration. It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet), .xml (OpenVINO).
    	--classes
    		Optional path to a text file with names of classes to label detected objects.
    	--device (value:0)
    		camera device number.
    	-f, --framework
    		Optional name of an origin framework of the model. Detect it automatically if it does not set.
    	-h, --help
    		Print help message.
    	--height (value:-1)
    		Preprocess input image by resizing to a specific height.
    	-i, --input
    		Path to input image or video file. Skip this argument to capture frames from a camera.
    	-m, --model
    		Path to a binary file of model contains trained weights. It could be a file with extensions .caffemodel (Caffe), .pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet), .bin (OpenVINO).
    	--mean
    		Preprocess input image by subtracting mean values. Mean values should be in BGR order and delimited by spaces.
    	--nms (value:.4)
    		Non-maximum suppression threshold.
    	--rgb
    		Indicate that model works with RGB input images instead BGR ones.
    	--scale (value:1.0)
    		Preprocess input image by multiplying on a scale factor.
    	--target (value:0)
    		Choose one of target computation devices: 0: CPU target (by default), 1: OpenCL, 2: OpenCL fp16 (half-float precision), 3: VPU
    	--thr (value:.5)
    		Confidence threshold.
    	--width (value:-1)
    		Preprocess input image by resizing to a specific width.
    	--zoo (value:models.yml)
    		An optional path to file with preprocessing parameters
    
    	alias
    		An alias name of model to extract preprocessing parameters from models.yml file.
