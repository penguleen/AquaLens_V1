import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Upload detector weights
model_path = os.path.join('.', 'runs', 'detect', 'train22', 'weights', 'best.pt')
model = YOLO(model_path)    #change
names = {0: 'fish'}     #change
VIDEOS_DIR = os.path.join('.')
frame_num = 0

# input video of fish/fishes
video_path = os.path.join(VIDEOS_DIR, 'light.mp4')        #edit

def overallfinal_video(video_path, model, names):
    video_path_out = 'finaloutput.mp4'.format(video_path)
    video_writers = {}

    cap = cv2.VideoCapture(video_path)
    ret, im = cap.read()
    H, W, _ = im.shape

    frame_num = 0

    #get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    new_fps = fps // 5  # Reduce the frame rate by half

    # Write the output video with the new frame rate = slow down output video
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), new_fps, (W, H))

    with open('finalname.txt', 'r') as file1, open('bbox.txt', 'r') as file2:
        # Read lines from both files
        bbox_name_file = file1.readlines()
        bbox_file = file2.readlines()

    # Zip the lines from both files together: fish labels and its respective bounding boxes
    paired_data = zip(bbox_name_file, bbox_file)

    # Iterate over the paired data and store it in a list
    bbox_data_list = list(paired_data)

    while cap.isOpened():

        success, frame = cap.read()
        if success:
            frame_num += 1
            results = model.track(frame, persist=True,show=True, tracker="botsort.yaml")

            boxes = results[0].boxes.xywh.cpu()  #bounding box
            clss = results[0].boxes.cls.cpu().tolist()  #class
            conf = results[0].boxes.conf.cpu().tolist()    #confidence score

            annotator = Annotator(frame, line_width=5,
                                      example=str(names))
            for box, cls, conf in zip(boxes, clss, conf):
                x, y, w, h = box
                x1, y1, x2, y2 = (x - w / 2, y - h / 2,
                                  x + w / 2, y + h / 2)

                # Iterate over the paired data and print it
                for pair in bbox_data_list:
                    track_id_from_file = pair[0].strip()  # Remove any excess information from fish label
                    bbox_data = pair[1].strip()  # Remove any excess information from bounding box label

                    # Parse the bounding box data string into a tuple of float values
                    bbox_values = tuple(map(float, bbox_data.strip('()').split(',')))

                    # Replace the bounding box values obtained from the model with the values from the file
                    x1_file, y1_file, x2_file, y2_file = bbox_values

                    # Link results (fish names) with the tracker's predicted bounding box
                    if x1 == x1_file and y1 == y1_file and x2 == x2_file and y2 == y2_file:
                        label = str(names[cls]) + " : " + str(track_id_from_file)
                        annotator.box_label([x1, y1, x2, y2], label, (218, 100, 255))
                        break  # Exit the inner loop once the label is displayed

            # Display frame number on top left corner of output video
            cv2.putText(frame, str(frame_num), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            cv2.imshow("YOLOv8 Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                out.write(frame)
                break
            out.write(frame)
            success, frame = cap.read()
        else:
            break

    for video_writer in video_writers.values():
        video_writer.release()

    cap.release()
    out.release()     # no output video detection shown. figure it out
    cv2.destroyAllWindows()

overallfinal_video(video_path, model, names)