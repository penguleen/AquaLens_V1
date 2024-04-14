import cv2
import os

def record_video(directory, filename, duration):
    os.makedirs(directory, exist_ok=True)

    # Construct the full path for the video file
    filepath = os.path.join(directory, filename)
    # Open the camera
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # You may need to adjust the camera index if necessary

    # Set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Width in pixels
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Height in pixels

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filepath, fourcc, 120.0, (1280, 720))  # Adjust resolution if needed

    # Record video for the specified duration
    start_time = cv2.getTickCount()
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < duration:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break

            # Write the frame to the video file
        out.write(frame)

            # Display the frame
        cv2.imshow('Recording', frame)

            # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Release the camera and video writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()


