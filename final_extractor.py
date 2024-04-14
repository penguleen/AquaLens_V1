import os
import cv2
from pathlib import Path
from ultralytics import YOLO

#Upload detector weights
model_path = os.path.join('.', 'runs', 'detect', 'train22', 'weights', 'best.pt')
model = YOLO(model_path)    #change
VIDEOS_DIR = os.path.join('.')

#input video of fish/fishes
video_path = os.path.join(VIDEOS_DIR, 'light.mp4')        #edit
def final_extraction(video_path, model):
    video_path_out = 'finaloutput.mp4'.format(video_path)
    updated_bbox = []
    video_writers = {}

    cap = cv2.VideoCapture(video_path)
    ret, im = cap.read()
    H, W, _ = im.shape

    frame_num = 0
    bb_list = []

    #get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    new_fps = fps // 5  # Reduce the frame rate by half
    width_frame = int(240)        #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_frame = int(135)      #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Write the output video with the new frame rate

    with open('finalname.txt', 'r') as file1, open('bbox.txt', 'r') as file2:
        # Read lines from both files
        bbox_name_file = file1.readlines()
        bbox_file = file2.readlines()

        # Zip the lines from both files together
    paired_data = zip(bbox_name_file, bbox_file)

    # Iterate over the paired data and store it in a list
    bbox_data_list = list(paired_data)

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Source path "
                                f"'{video_path}' "
                                f"does not exist.")
    def get_bbox_and_id(track_id,bbox):
        bounding_boxes=[]
        bounding_boxes.append({"track_id": track_id, "bbox":bbox})
        bbox_string = ', '.join(map(str, bbox))
        bbox_stuff = str(track_id)+','+str(bbox_string) +'\n'

        return bounding_boxes

    while cap.isOpened():

        success, frame = cap.read()
        if success:
            #frame_num += 1
            results = model.track(frame, persist=True, show=True, tracker="botsort.yaml")

            boxes = results[0].boxes.xywh.cpu()  #bounding box
            clss = results[0].boxes.cls.cpu().tolist()  #class
            conf = results[0].boxes.conf.cpu().tolist()    #confidence score

            for box, cls, conf in zip(boxes, clss, conf):
                x, y, w, h = box
                x1, y1, x2, y2 = (x - w / 2, y - h / 2,
                                  x + w / 2, y + h / 2)
                    # Iterate over the paired data and print it
                for pair in bbox_data_list:
                    track_id_from_file = pair[0].strip()  # Remove any leading or trailing whitespace
                    bbox_data = pair[1].strip()  # Remove any leading or trailing whitespace

                    # Parse the bounding box data string into a tuple of float values
                    bbox_values = tuple(map(float, bbox_data.strip('()').split(',')))

                    # Replace the bounding box values obtained from the model with the values from the file
                    x1_file, y1_file, x2_file, y2_file = bbox_values

                    if x1 == x1_file and y1 == y1_file and x2 == x2_file and y2 == y2_file:
                        for item in get_bbox_and_id(track_id_from_file, bbox_values):
                            # outcome = extract_and_pair_data(file_bbox_path)  # creates a dictionary
                            # create_files(file_bboxdata, outcome)
                            track_id = item["track_id"]
                            bbox = item["bbox"]
                            width = bbox[2] - bbox[0]
                            height = bbox[3] - bbox[1]

                            # Create respective video files for respective unique id
                            if track_id not in video_writers:
                                output_video_path = f'output_id_{track_id}.avi'
                                video_writers[track_id] = cv2.VideoWriter(output_video_path,
                                                                          cv2.VideoWriter_fourcc(*'XVID'), new_fps,
                                                                          (width_frame, height_frame))

                            # Resize the frame to match the bounding box dimensions
                            cropped_frame = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                            full_screen_frame = cv2.resize(cropped_frame, (width_frame, height_frame))

                            # Write the resized frame to the VideoWriter
                            video_writers[track_id].write(full_screen_frame)
                        break  # Exit the inner loop once the label is displayed

            cv2.imshow("YOLOv8 Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            success, frame = cap.read()
        else:
            break

    for video_writer in video_writers.values():
        video_writer.release()

    cap.release()
    cv2.destroyAllWindows()

final_extraction(video_path, model)