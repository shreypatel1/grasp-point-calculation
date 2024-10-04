from ultralytics import YOLO
import cv2
import numpy as np
import time
import pyrealsense2 as rs

source = 0
model = YOLO('yolo11n-seg.pt')

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

time.sleep(2)

# Check if any device is connected
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("No device connected")
    exit(0)
else:
    print("Connected devices:")
    print(f"Number of devices: {len(devices)}")
    print(devices)
    for i, device in enumerate(devices):
        print(f"Device {i}: {device.get_info(rs.camera_info.name)}")

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 25)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 25)

# Start streaming
pipeline.start(config)

while True:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames(10000)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Run YOLO model on the color image
    results = model(color_image, show=True, conf=0.4)

    # Iterate through results and print them
    for result in results:
        # Print predictions in pandas format
        print(result.pandas().xyxy[0])  # Access the first image's predictions

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((color_image, depth_colormap))

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(1)

    # Press esc or 'q' to close the image window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop streaming
pipeline.stop()
cv2.destroyAllWindows()
