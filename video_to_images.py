import supervision as sv

def convertvideo_to_images(VIDEO_DIR_PATH, IMAGE_DIR_PATH, videonumber):

    FRAME_STRIDE = 10       #changes every 10th frame to image


    video_paths = sv.list_files_with_extensions(
        directory= VIDEO_DIR_PATH,
        extensions=["mov", "mp4"])

    TEST_VIDEO_PATHS, TRAIN_VIDEO_PATHS = video_paths[: int(videonumber)], video_paths[int(videonumber):]  #get from the first video in the video_obtained folder, for testing videonumber=0


    for video_path in TRAIN_VIDEO_PATHS:
        video_name = video_path.stem
        print(video_name)
        image_name_pattern = video_name + "-{:05d}.png"
        with sv.ImageSink(target_dir_path=IMAGE_DIR_PATH, image_name_pattern=image_name_pattern) as sink:
            for image in sv.get_video_frames_generator(source_path=str(video_path), stride=FRAME_STRIDE):
                sink.save_image(image=image)

    image_paths = sv.list_files_with_extensions(
        directory=IMAGE_DIR_PATH,
        extensions=["png", "jpg", "jpg"])

    print('image count:', len(image_paths))

    return image_paths

