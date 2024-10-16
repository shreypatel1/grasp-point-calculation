import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# config.enable_stream(rs.stream.depth, 640, 420, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 416, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 416, rs.format.y8, 30)
config.enable_stream(rs.stream.depth, 640, 416, rs.format.z16, 30)

# Start streaming
try:
    pipeline.start(config)
except RuntimeError as e:
    print("RuntimeError:", e)

while True:
    # Wait for frames from the Realsense Camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    infrared_frame = frames.get_infrared_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame:
        print("No frame received!")
        break

    # Convert images to numpy arrays
    color_frame = np.asanyarray(color_frame.get_data())
    infrared_frame = np.asanyarray(infrared_frame.get_data())
    depth_frame = np.asanyarray(depth_frame.get_data())

    # Display the infrared as an array of numbers
    # print(infrared_frame)

    cv2.imshow("Color Frame", color_frame)
    cv2.imshow("Infrared Frame", infrared_frame)
    cv2.imshow("Depth Frame", depth_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break
