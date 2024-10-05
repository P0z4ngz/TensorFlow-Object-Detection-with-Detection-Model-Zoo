# File to run

from detection import Detector

# Based on Tensorflow 2 Detection Model ZOO
# You can get the pre-trained model on https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz"

IMG_PATH = "pictures/motogp_dataset.jpg"  #customizable
VID_PATH = "videos/traffic_highways.mp4"    ##customizable
CLASS_PATH = "object_class_names.txt"   # From COCO dataset

detector = Detector()
detector.read_classes(CLASS_PATH)
detector.get_model(MODEL_URL)
detector.load_model()
detector.predict_image(image_path=IMG_PATH)

# Comment the detector.predict_image() and Uncomment this to run object detection video
#detector.predict_video(video_path=VID_PATH)