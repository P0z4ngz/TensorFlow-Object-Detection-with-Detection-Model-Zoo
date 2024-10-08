# TensorFlow Object Detection with Detection Model Zoo

## Description
This repository contains an object detection project utilizing pre-trained models from the [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), specifically trained on the [COCO 2017 dataset](http://cocodataset.org/). The project includes functionality to detect objects in both images and videos.

## Project Overview
This project implements an object detection system based on **TensorFlow 2** and pre-trained models from the TensorFlow 2 Detection Model Zoo. The system can:
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

## Log
