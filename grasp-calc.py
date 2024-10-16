import argparse
from ultralytics import YOLO
import cv2
import numpy as np
from imutils.video import VideoStream, FPS
from realsense import RealSenseCamera


# Construct an argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--source", "-s", type=int, default=0, help="camera source")
ap.add_argument("--weights", "-w", type=str, default="runs/segment/300epochs/weights/best.pt", help="YOLO segmentation model weights")
ap.add_argument("--conf", "-c", type=float, default=0.90, help="confidence threshold")
args = ap.parse_args()

# Initialize YOLO model
model = YOLO(args.weights)

# Testing purposes: Initialize the video stream and FPS counter
# vs = VideoStream(sec=args.source).start()
fps = FPS().start()

# Initialize the RealSense camera
rs_cam = RealSenseCamera()

# Loop over the frames from the video stream
while True:
    has_frames, depth_frame, color_frame = rs_cam.get_frames()
    if not has_frames:
        break
    # depth_frame = cv2.resize(frame, (640, 416))
    # color_frame = cv2.resize(frame, (640, 416))

    # Run the YOLO model on the frame
    results = model(color_frame, conf=args.conf, stream=True)

    for r in results: # results is a generator
        for idx, mask in enumerate(r.masks.data):

            if len(mask) > 0 and len(depth_frame) > 0:
                # Extract the mask and convert it to a NumPy array
                mask = mask.cpu().numpy().astype(np.uint8)
                mask_binary = (mask > 0).astype(np.uint8)
                print(mask_binary)

                # Apply the mask to the color frame and isolate the object
                isolated_color_frame = cv2.bitwise_and(color_frame, color_frame, mask=mask_binary)

                # Get the bounding box of the object for isolating the depth frame
                bbox = r.boxes.data[idx].cpu().numpy().astype(np.float16)
                print(bbox)
                x1, y1, x2, y2 = int(bbox)
                print(x1, y1, x2, y2)
                isolated_color_frame = isolated_color_frame[y1:y2, x1:x2]
                mask_binary = mask_binary[y1:y2, x1:x2]
                isolated_depth_frame = depth_frame[y1:y2, x1:x2]
                print(isolated_depth_frame)

                # Find the non-zero pixels in the mask binary and convert them to an array of (x, y) coordinates
                nonzero = np.argwhere(mask_binary)
                print(nonzero)
                distance_data = []
                for x, y in nonzero:
                    distance_data.append([x, y, isolated_depth_frame[y, x]])
                distance_data = np.array(distance_data)
                print(distance_data)

                if len(depth_frame) > 0:
                    # Display the isolated color frame
                    cv2.namedWindow("Isolated Color Frame", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("Isolated Color Frame", isolated_color_frame)
                    cv2.namedWindow("Isolated Depth Frame", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("Isolated Depth Frame", isolated_depth_frame)

    # Show the original color frame for reference
    cv2.imshow("Color Frame", color_frame)

    # Update the FPS counter
    fps.update()

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break
