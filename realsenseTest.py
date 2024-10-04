import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable depth and color streams
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 25)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 25)

try:
    # Start streaming
    pipeline.start(config)

    while True:
        # Wait for frames from the RealSense camera
        frames = pipeline.wait_for_frames(10000)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Display the images
        cv2.imshow('Depth Image', depth_image)
        cv2.imshow('Color Image', color_image)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()

cv2.destroyAllWindows()
