# TensorFlow Object Detection with Detection Model Zoo

## Description
This repository contains an object detection project utilizing pre-trained models from the [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), specifically trained on the [COCO 2017 dataset](http://cocodataset.org/) which contains 80 object categories such as people, animals, and everyday objects. The project includes functionality to detect objects in both images and videos.

## Project Overview
This project implements an object detection system based on **TensorFlow 2** and pre-trained models from the TensorFlow 2 Detection Model Zoo which includes various neural network architectures that have been optimized for object detection with varying performance based on mAP (mean Average Precision) and inference speed . The system can:
* Load and utilize a model from the Detection Model Zoo.
* Detect objects in images and videos.
* Display bounding boxes with labels and confidence scores around detected objects.

The project consists of two main Python files:
1. `detection.py` - Contains the Detector class which includes the following methods:
   * `read_classes` : Reads object classes from a file.
   * `get_model` : Downloads the pre-trained model.
   * `load_model` : Loads the downloaded model for inference.
   * `bounding_box` : Draws bounding boxes on detected objects.
   * `predict_image` : Detects objects in an image.
   * `predict_video` : Detects objects in a video.

2. `main.py` - Runs the detection using variables:
   * `MODEL_URL` : URL of the [TensorFlow 2 Detection Model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) (zip file link).
   * `IMG_PATH` : Path to the image file for detection (customizable).
   * `VID_PATH` : Path to the video file for detection (customizable).
   * `CLASS_PATH` : Path to the classes file (COCO labels).

## Usage
1. Download the desired model from [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and change the **MODEL_URL** in `main.py`.

2. Update the paths in `main.py`:
  * **IMG_PATH**: Path to the image file for object detection.
  * **VID_PATH**: Path to the video file for object detection.

3. Run the `main.py` to perform detection:
```
python3 main.py
```

#### Note
You can explore various models with different trade-offs between accuracy and speed, including:
  * [Faster R-CNN](https://medium.com/thedeephub/faster-r-cnn-object-detection-5dfe77104e31)
  * [SSD MobileNet](https://medium.com/@tauseefahmad12/object-detection-using-mobilenet-ssd-e75b177567ee)
  * [EfficientDet](https://medium.com/@vipas.ai/efficientdet-a-powerful-object-detection-model-50b5ae10113f)
  * [CenterNet](https://medium.com/visionwizard/centernet-objects-as-points-a-comprehensive-guide-2ed9993c48bc)
