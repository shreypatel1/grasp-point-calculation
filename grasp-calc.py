import argparse

from torchgen.gen import format_yaml
from ultralytics import YOLO
import cv2
import numpy as np
from imutils.video import FPS
from realsense import RealSenseCamera
import open3d as o3d


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

x_scale = 1280 / 640
y_scale = 720 / 384

# Create a point cloud object
point_cloud = o3d.geometry.PointCloud()
# Start visualizing point cloud in a separate thread
vis = o3d.visualization.Visualizer()
vis.create_window("Point Cloud")

intrinsic = rs_cam.get_intrinsics()
# [ 1280x720  p[654.9 363.624]  f[643.633 643.024]  Inverse Brown Conrady [-0.0544022 0.0635166 -0.000826826 0.000847402 -0.0205106] ]
fx, fy = intrinsic.fx, intrinsic.fy
cx, cy = intrinsic.ppx, intrinsic.ppy
k1, k2, p1, p2, k3 = intrinsic.coeffs

# Camera matrix
camera_matrix = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]], dtype=np.float32)

# Distortion coefficients
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)


# Loop over the frames from the video stream
while True:
    has_frames, depth_frame, color_frame = rs_cam.get_frames()
    if not has_frames:
        break
    # depth_frame = cv2.resize(frame, (640, 384))
    # color_frame = cv2.resize(frame, (640, 384))

    # Undistort the frames
    color_frame = cv2.undistort(color_frame, camera_matrix, dist_coeffs)
    depth_frame = cv2.undistort(depth_frame, camera_matrix, dist_coeffs)

    # Run the YOLO model on the frame
    results = model(color_frame, conf=args.conf, stream=True)

    for r in results: # results is a generator
        if r.masks is None or len(r.masks.data) == 0:
            break
        for idx, mask in enumerate(r.masks.data):

            if len(mask) > 0 and len(depth_frame) > 0:
                # Extract the mask and convert it to a NumPy array
                mask = mask.cpu().numpy().astype(np.uint8)
                mask_binary = (mask > 0).astype(np.uint8)
                # convert mask binary to the same size as the depth and color frames
                mask_binary = cv2.resize(mask_binary, (1280, 720))
                print(f"Mask Binary size: {len(mask_binary[0])} x {len(mask_binary)}")
                print(f"Depth Frame size: {len(depth_frame[0])} x {len(depth_frame)}")
                print(f"Color Frame size: {len(color_frame[0])} x {len(color_frame)}")

                # Apply the mask to the color frame and isolate the object
                isolated_color_frame = cv2.bitwise_and(color_frame, color_frame, mask=mask_binary)
                cv2.imshow("Whole Isolated Color Frame", isolated_color_frame)

                # Get the bounding box of the object for isolating the depth frame
                bbox = r.boxes.data[idx].cpu().numpy()
                x1, y1, x2, y2, _, _ = bbox.astype(np.uint32)
                # x1, y1, x2, y2 = int(x1 * x_scale), int(y1 * y_scale), int(x2 * x_scale), int(y2 * y_scale)
                # print(x1, y1, x2, y2)
                # print(y2-y1, x2-x1)
                isolated_color_frame = isolated_color_frame[y1:y2, x1:x2]
                mask_binary = mask_binary[y1:y2, x1:x2]
                isolated_depth_frame = depth_frame[y1:y2, x1:x2]
                # print(f"Isolated Mask Binary size: {len(mask_binary)} x {len(mask_binary[0])}")
                # print(f"Isolated Depth Frame size: {len(isolated_depth_frame)} x {len(isolated_depth_frame[0])}")

                points = []
                # colors = []

                # Find the non-zero pixels in the mask binary and convert them to an array of (x, y) coordinates
                nonzero = np.argwhere(mask_binary)
                # print(nonzero)
                distance_data = []
                for y, x in nonzero:
                    Z = isolated_depth_frame[y, x]
                    if Z > 0:
                        X = ((x - cx) * Z) / fx
                        Y = ((y - cy) * Z) / fy
                        points.append([X, Y, Z])
                        # colors.append(isolated_color_frame[y, x] / 255.0)
                        distance_data.append([x, y, isolated_depth_frame[y, x]])
                distance_data = np.array(distance_data)
                # print(distance_data)

                # if point_cloud.shape[0] > 0:
                #     pcd = o3d.geometry.PointCloud()
                #     pcd.points = o3d.utility.Vector3dVector(point_cloud)
                #     o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud")

                print(f"Total points extracted: {len(points)}")
                if len(points) > 0:
                    print(f"Sample points: {points[:5]}")  # Print the first 5 points
                else:
                    print("No valid points extracted.")

                if len(points) > 0:
                    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
                    vis.clear_geometries()  # Clear previous geometries
                    vis.add_geometry(point_cloud)
                    vis.update_geometry(point_cloud)
                    vis.poll_events()
                    vis.update_renderer()


                if len(depth_frame) > 0:
                    # Display the isolated color frame
                    cv2.namedWindow("Isolated Color Frame", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("Isolated Color Frame", isolated_color_frame)
                    # cv2.namedWindow("Isolated Depth Frame", cv2.WINDOW_AUTOSIZE)
                    # cv2.imshow("Isolated Depth Frame", isolated_depth_frame)

    # Show the original color frame for reference
    # cv2.imshow("Color Frame", color_frame)

    # Update the FPS counter
    fps.update()

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break
