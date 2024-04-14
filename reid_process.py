# Tracking and re-identification after classification model is trained and object detection model weights is created.
import cv2
from pathlib import Path
from ultralytics import YOLO
import os
import tensorflow as tf

# Imported python files
from video_to_frames import convertvideo_to_images
from wave1_labels import classification
from label2_extraction import extraction2

file_bbox = open("bbox.txt", "w")       # collect all predicted bounding boxes from tracker

boundingbox_dict = {}
original_dict = {}
scores_dict = {}
tested_dict = {}
recheckpath_dict = {}
recheckoutput_dict = {}
newscores_dict = {}
video_writers = {}

# Create a list to store tested_paths for lines with all scores < 80%
path_for_line_list=[]
bbox_list=[]

#Upload detector weights
model_path = os.path.join('.', 'runs', 'detect', 'train22', 'weights', 'best.pt')  # object detection weights
model = YOLO(model_path)    # object detection model
names = {0: 'fish'}     # class within the trained model
VIDEOS_DIR = os.path.join('.')

#input video of fish/fishes
video_path = os.path.join(VIDEOS_DIR, 'light.mp4')     # input fish video
updated_bbox = []

cap = cv2.VideoCapture(video_path)
ret, im = cap.read()
H, W, _ = im.shape
frame_num = 0
bb_list = []

track_id = 1

#get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width_frame = int(240)      # new width frame
height_frame = int(135)      # new height frame

# image paths for classification purpose
ideal_fish_data= Path('./ideal_class_images')
checkingall_fishes = Path('./ideal_class_images/all')

all_ideals = list(ideal_fish_data.glob('all/*'))        # used to compare extracted image to 100 images of all 5 fishes
all_ideals_path = [str(path) for path in all_ideals]

fish_data= os.path.join('.', '5classes')        # images of all the 5 fishes trained in classification model

training_batch_size=32

train_set = tf.keras.preprocessing.image_dataset_from_directory(  # train settings for classification model deployment
  fish_data,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(height_frame,width_frame),
  batch_size=training_batch_size)

validation_set = tf.keras.preprocessing.image_dataset_from_directory(   # validation settings for classification model deployment
  fish_data,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(height_frame, width_frame),
  batch_size=training_batch_size)

image_cat = train_set.class_names    # classes trained by classification model

# Collect all the unique ids and corresponding bounding boxes
def get_bbox_and_id(track_id,bbox):
    bounding_boxes= []
    bounding_boxes.append({"track_id": track_id, "bbox": bbox})

    return bounding_boxes

while cap.isOpened():

    success, frame = cap.read()
    if success:
        frame_num += 1
        results = model.track(frame, persist=True, show=True, tracker="botsort.yaml") # added botsort tracker

        boxes = results[0].boxes.xywh.cpu()  # predicted bounding box
        clss = results[0].boxes.cls.cpu().tolist()  # class
        conf = results[0].boxes.conf.cpu().tolist()    # confidence score
        print(frame_num)

        for box, cls, conf in zip(boxes, clss, conf):
            x, y, w, h = box
            x1, y1, x2, y2 = (x - w / 2, y - h / 2,
                              x + w / 2, y + h / 2)

            x1 = (x1.item())  # extract tensor number and convert to numerical
            y1 = (y1.item())
            x2 = (x2.item())
            y2 = (y2.item())

            bbox = (x1, y1, x2, y2)

            bb_list.append(bbox)
            bbox_data = str(bbox) + '\n'
            file_bbox.write(bbox_data)  # collect all predicted bounding boxes in textfile

            # Obtain all height and width of the predicted bounding boxes
            for item in get_bbox_and_id(track_id, bbox):
                track_id = item["track_id"]
                bbox = item["bbox"]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                # Create respective video files for respective unique id
                if track_id not in video_writers:
                    output_video_path = f'./videofile/_output.avi'
                    video_writers[track_id] = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width_frame, height_frame))

                # Resize the frame to match the bounding box dimensions
                cropped_frame = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                full_screen_frame = cv2.resize(cropped_frame, (width_frame, height_frame))

                # Write the resized frame to the VideoWriter
                video_writers[track_id].write(full_screen_frame)

        cv2.imshow("YOLOv8 Detection", frame)       # Display output of tracker

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        success, frame = cap.read()
    else:
        break

for video_writer in video_writers.values():
    video_writer.release()

cap.release()
cv2.destroyAllWindows()

VIDEO_DIR_PATH = os.path.join(".","videofile")
IMAGE_DIR_PATH = os.path.join(".","video_frames")
convertvideo_to_images(VIDEO_DIR_PATH, IMAGE_DIR_PATH, 0)

collected_fish_data= Path('./video_frames')
tested_fish = list(collected_fish_data.glob('*'))

classification(tested_fish, image_cat)             # Send for classification to extract 1st round of fish labels
extraction2(all_ideals_path, all_ideals, ideal_fish_data, tested_fish)     # Send for feature extraction to filter 2st round of fish labels
#overallfinal_video(video_path, model, names)
#final_extraction(video_path, model)
