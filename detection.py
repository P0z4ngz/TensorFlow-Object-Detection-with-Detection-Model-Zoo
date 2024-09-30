# Object detections class

import cv2 as cv
import time, os, random
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import get_file # type: ignore

#---------------------------------------------------------------------------------------------------------#
tf.random.set_seed(42)

class Detector:
  def __init__(self):
      pass
  
  # Open a class names model file
  def read_classes(self, class_path):
      with open(class_path, 'r') as f:
        self.class_names = f.read().splitlines()
        
      # Define color list
      self.color_list = np.random.uniform(low=0, high=255, size=(len(self.class_names), 3))
  
  # Get pre-trained model
  def get_model(self, model_url):
     model_file = os.path.basename(model_url)
     self.model_name = model_file[:model_file.index('.')]

     # Download pretrained model
     get_file(fname=model_file, cache_dir='saved_models', origin=model_url, cache_subdir='checkpoints', extract=True)
     
  # Load a model
  def load_model(self):   
     self.model = tf.saved_model.load(os.path.join('saved_models', 'checkpoints', self.model_name, 'saved_model'))
     print(f"Model {self.model_name} loaded successfully...\n")

  # Create Bounding Box   
  def bounding_box(self, img_tensor, threshold=0.5):
     input_tensor = cv.cvtColor(img_tensor.copy(), cv.COLOR_BGR2RGB)
     input_tensor = tf.expand_dims(tf.convert_to_tensor(input_tensor, dtype=tf.uint8), axis=0)

     # Model detection processing
     detections = self.model(input_tensor)
     bboxs = detections['detection_boxes'][0].numpy()
     class_indexes = detections['detection_classes'][0].numpy().astype(np.int32)
     class_scores = detections['detection_scores'][0].numpy()
     
     # Set image height, width, and channel from img_tensor.shape
     imgH, imgW, imgC = img_tensor.shape
     # Set threshold of bounding box
     bbox_index = tf.image.non_max_suppression(bboxs, scores=class_scores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)
     
     # Iterate throughout bbox_index
     if len(bbox_index) != 0:
      for i in bbox_index:
         bbox = tuple(bboxs[i].tolist())
         class_confidence = round(100 * class_scores[i])
         class_index = class_indexes[i]

         class_label = self.class_names[class_index].upper()
         class_color = self.color_list[class_index]
         display_text = f"{class_label} {class_confidence}%"
         
         # Bounding box scale
         ymin, xmin, ymax, xmax = bbox
         ymin, xmin, ymax, xmax = (ymin * imgH, xmin * imgW, ymax * imgH, xmax * imgW)
         ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)

         cv.rectangle(img_tensor, (xmin, ymin), (xmax, ymax), color=class_color, thickness=2)
         cv.putText(img_tensor, display_text, (xmin, ymin - 10), cv.FONT_HERSHEY_PLAIN, 1, class_color, 2)

      return img_tensor

  # Predict Image function
  def predict_image(self, image_path, threshold=0.5):
     image_tensor = cv.imread(image_path)
     bbox_img = self.bounding_box(img_tensor=image_tensor, threshold=threshold)
     img_save_path = "model_results/pictures/"  #save directory path

     # Save result
     cv.imwrite(img_save_path + self.model_name + "-" + image_path.split('/')[1], bbox_img)
     cv.imshow("Result" ,bbox_img)
     cv.waitKey(0)
     cv.destroyAllWindows()

  # Predict Video function
  def predict_video(self, video_path):
     cap = cv.VideoCapture(video_path)
     
     # Check if the video cannot open
     if cap.isOpened == False:
        return print("Error open a video....")
     
     # Read the first frame
     (success, vid_tensor) = cap.read()  
     
     # Set FPS
     start_time = 0
     while success:
         current_time = time.time()
         fps = 1 / (current_time - start_time)
         start_time = current_time

         bbox_vid = self.bounding_box(img_tensor=vid_tensor)
         cv.putText(bbox_vid, "FPS: " + str(int(fps)), (20, 70), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
         cv.imshow("Result", bbox_vid)
  
         # Exit loop condition
         key = cv.waitKey(1) & 0xFF
         if key == ord("q"):
            break

         (success, vid_tensor) = cap.read()
     cv.destroyAllWindows()
