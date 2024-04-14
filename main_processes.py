import supervision as sv
from autodistill.detection import CaptionOntology
import cv2
from autodistill_grounded_sam import GroundedSAM
from video_to_images import convertvideo_to_images
import os
from camera import record_video
from yolov8main import yolov8_training

##press of a button will run
#start by recording a video
directory = './videos_obtained'

num_of_videos = int(input("Enter the number of 20 min videos for training: "))

# start camera, recording with camera for x amount of time
# save footage of camera in the name:"capturedvideo_1" in .mp4 format into videos_obtained folder
# remember the number of the captured video

record_timer = int(20*60)       # 20 minutes recording
for i in range(num_of_videos):
    filename = f"capturedvideo_{i + 1}.avi"
    print(f"Recording video {i + 1}...")
    record_video(directory, filename, record_timer)
    print(f"Video {i + 1} recorded.")

print("All videos recorded.")

# create a folder: video, convert the video into images #count the amount of images converted
VIDEO_DIR_PATH = os.path.join(".","videos_obtained")
IMAGE_DIR_PATH = os.path.join(".","images_converted")

image_paths = convertvideo_to_images(VIDEO_DIR_PATH, IMAGE_DIR_PATH,2)   # convert input videos to images
SAMPLE_SIZE = 16
SAMPLE_GRID_SIZE = (4, 4)
SAMPLE_PLOT_SIZE = (16, 16)

titles = [
    image_path.stem
    for image_path
    in image_paths[:SAMPLE_SIZE]]
images = [
    cv2.imread(str(image_path))
    for image_path
    in image_paths[:SAMPLE_SIZE]]

#define ontology from the autolabel dataset
# This base model will create files: annotation, images, train, vaild and data.yaml after processing all the images

ontology=CaptionOntology({
    "fish": "fish"
})
DATASET_DIR_PATH = os.path.join(".","dataset")

# initiate base model and autolabel
base_model = GroundedSAM(ontology=ontology)
dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH,
    extension=".png",
    output_folder=DATASET_DIR_PATH)

#after it runs:
#display dataset sample

ANNOTATIONS_DIRECTORY_PATH = os.path.join(".","dataset", "train", "labels")
IMAGES_DIRECTORY_PATH = os.path.join(".","dataset", "train", "images")
DATA_YAML_PATH = os.path.join(".","dataset", "data.yaml")

dataset_1 = sv.DetectionDataset.from_yolo(
        images_directory_path=IMAGES_DIRECTORY_PATH,
        annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
        data_yaml_path=DATA_YAML_PATH)

image_names = list(dataset_1.images.keys())[:SAMPLE_SIZE]

mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()

images1 = []
for name in image_names:
    image = dataset_1.images1[name]
    annotations = dataset_1.annotations[name]
    labels = [
        dataset_1.classes[class_id]
        for class_id
        in annotations.class_id]
    annotates_image = mask_annotator.annotate(
        scene=image.copy(),
        detections=annotations)
    annotates_image = box_annotator.annotate(
        scene=annotates_image,
        detections=annotations,
        labels=labels)
    images1.append(annotates_image)

# Transfer to Yolov8 for training, pre-set at 100 epoch
yolov8_training(DATA_YAML_PATH)


