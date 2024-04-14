import supervision as sv
import os

# Clear all previous images
def clear_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Remove all files in the directory
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    else:
        # If the directory doesn't exist, create it
        os.makedirs(directory_path)
def convertvideo_to_images(VIDEO_DIR_PATH, IMAGE_DIR_PATH, videonumber):

    FRAME_STRIDE = 0

    clear_directory(IMAGE_DIR_PATH)

    video_paths = sv.list_files_with_extensions(
        directory= VIDEO_DIR_PATH,
        extensions=["mov", "mp4", "avi"])

    TEST_VIDEO_PATHS, TRAIN_VIDEO_PATHS = video_paths[: int(videonumber)], video_paths[int(videonumber):]


    for video_path in TRAIN_VIDEO_PATHS:
        video_name = video_path.stem
        print(video_name)
        image_name_pattern = "fish_{:06d}.jpg"
        with sv.ImageSink(target_dir_path=IMAGE_DIR_PATH, image_name_pattern=image_name_pattern) as sink:
            frame_number = 1
            for image in sv.get_video_frames_generator(source_path=str(video_path), stride=FRAME_STRIDE):
                sink.save_image(image=image, image_name=image_name_pattern.format(frame_number))
                frame_number += 1  # Increment frame number

    image_paths = sv.list_files_with_extensions(
        directory=IMAGE_DIR_PATH,
        extensions=["png", "jpg", "jpg"])

    print('image count:', len(image_paths))     # Number of images which are the predicted bounding boxes collected

    return image_paths
