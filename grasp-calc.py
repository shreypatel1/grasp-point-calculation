import argparse

from ultralytics import YOLO
import cv2
import numpy as np
from imutils.video import FPS

import math_models
from realsense import RealSenseCamera
import open3d as o3d
from sklearn.decomposition import PCA

# Construct an argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--weights", "-w", type=str, default="runs/segment/300epochs/weights/best.pt",
                help="YOLO segmentation model weights")
ap.add_argument("--conf", "-c", type=float, default=0.90, help="confidence threshold")
args = ap.parse_args()

# Initialize YOLO model
model = YOLO(args.weights)

# Testing purposes: Initialize the video stream and FPS counter
fps = FPS().start()

# Initialize the RealSense camera
rs_cam = RealSenseCamera()

color_intrinsics = rs_cam.get_color_intrinsics()
# [ 1280x720  p[654.9 363.624]  f[643.633 643.024]  Inverse Brown Conrady [-0.0544022 0.0635166 -0.000826826 0.000847402 -0.0205106] ]
fx, fy = color_intrinsics.fx, color_intrinsics.fy
cx, cy = color_intrinsics.ppx, color_intrinsics.ppy

depth_intrinsics, depth_scale = rs_cam.get_depth_intrinsics()
extrinsics = rs_cam.get_extrinsics()
tx, ty, tz = extrinsics.translation
px_offset = int((tx * fx)) - 6
py_offset = int((ty * fy)) + 3
print(f"px_offset: {px_offset}, py_offset: {py_offset}")

# print(f"Color Intrinsics: {color_intrinsics}")
# print(f"Depth Intrinsics: {depth_intrinsics}")
# print(f"Extrinsics: {extrinsics}")

# Camera matrix and distortion coefficients
cam_intr = math_models.CameraIntrinsics(fx, fy, cx, cy, color_intrinsics.coeffs)
camera_matrix = cam_intr.get_camera_matrix()
dist_coeffs = cam_intr.get_distortion_coeffs()

# Loop over the frames from the video stream
while True:
    has_frames, depth_frame, color_frame = rs_cam.get_frames()
    if not has_frames:
        break

    # Undistort the frames
    color_frame = cv2.undistort(color_frame, camera_matrix, dist_coeffs)
    depth_frame = cv2.undistort(depth_frame, camera_matrix, dist_coeffs)

    # Run the YOLO model on the frame
    results = model(color_frame, conf=args.conf, stream=True)

    for r in results:  # results is a generator
        if r.masks is None or len(r.masks.data) == 0:
            break
        for idx, mask in enumerate(r.masks.data):

            if len(mask) > 0 and len(depth_frame) > 0:
                # Extract the mask and convert it to a NumPy array
                mask = mask.cpu().numpy().astype(np.uint8)
                mask_binary = (mask > 0).astype(np.uint8)
                # convert mask binary to the same size as the depth and color frames
                mask_binary = cv2.resize(mask_binary, (1280, 720))
                # print(f"Mask Binary size: {len(mask_binary[0])} x {len(mask_binary)}")
                # print(f"Depth Frame size: {len(depth_frame[0])} x {len(depth_frame)}")
                # print(f"Color Frame size: {len(color_frame[0])} x {len(color_frame)}")

                # Apply the mask to the color frame and isolate the object
                isolated_color_frame = cv2.bitwise_and(color_frame, color_frame, mask=mask_binary)

                # Get the bounding box of the object for isolating the depth frame
                bbox = r.boxes.data[idx].cpu().numpy()
                x1, y1, x2, y2, _, _ = bbox.astype(np.int32)
                isolated_color_frame = isolated_color_frame[y1:y2, x1:x2]
                mask_binary = mask_binary[y1:y2, x1:x2]
                isolated_depth_frame = depth_frame[(y1 - py_offset):(y2 - py_offset), (x1 - px_offset):(x2 - px_offset)]
                # print(f"Cropped frame size: {len(isolated_color_frame[0])} x {len(isolated_color_frame)}")

                points = []

                # Find the non-zero pixels in the mask binary and convert them to an array of (x, y) coordinates
                nonzero = np.argwhere(mask_binary)
                for y, x in nonzero:
                    dist = isolated_depth_frame[y, x] / 1000
                    if dist > 0:
                        X = ((x1 + x - cx) * dist) / fx
                        Y = ((y1 + y - cy) * dist) / fy
                        Z = np.sqrt(dist ** 2 - X ** 2 - Y ** 2)
                        points.append([X, Y, Z])

                points = np.array(points)

                # Create an Open3D point cloud object
                pcd = o3d.geometry.PointCloud()

                if len(points) > 0:
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=120, std_ratio=0.5,
                                                                              print_progress=True)
                    # get the updated set of points as a numpy array
                    points = np.asarray(pcd.points)
                    # line_set = o3d.geometry.LineSet()


                    # RANSAC Cylinder Fitting
                    distance_threshold = 0.01
                    ransac_n = 500  # Minimum number of points to fit a cylinder model
                    num_iterations = 1000  # Number of RANSAC iterations

                    # Generate plane samples and their normal vectors to calculate the cylinder axis
                    mmdl = math_models.MathModels()
                    plane_samples = mmdl.generate_plane_samples(points, num_samples=ransac_n, max_distance=distance_threshold)

                    # Add lines representing the plane normals
                    # line_pts = []
                    # lineset_lines = []
                    displacements = []
                    normals = []

                    for sample in plane_samples:
                        p1, p2, p3 = sample
                        center = (p1 + p2 + p3) / 3
                        normal = np.cross(p2 - p1, p3 - p1)
                        normal /= np.linalg.norm(normal)

                        # Ensure the normal vector points away from the origin
                        if np.dot(center, normal) < 0:
                            normal = -normal

                        displacements.append(center)
                        normals.append([normal * 0.05, center])

                        # line_pts.append(center)
                        # line_pts.append(center + normal * 0.05)  # Scale the normal for visualization
                        # lineset_lines.append([len(line_pts) - 2, len(line_pts) - 1])

                    # line_pts = np.array(line_pts)
                    # lineset_lines = np.array(lineset_lines, dtype=np.int32)
                    normals = np.array(normals)

                    # Add the points and lines to the line_set object
                    # line_set.points = o3d.utility.Vector3dVector(line_pts)
                    # line_set.lines = o3d.utility.Vector2iVector(lineset_lines)

                    # Calculate the cylinder axis line using the plane normals and least squares fitting
                    pca = PCA(n_components=1)
                    pca.fit(displacements)
                    initial_axis = pca.components_[0]
                    axis_point = np.mean(displacements, axis=0)

                    axis = initial_axis
                    for _ in range(num_iterations):  # Iterate to refine the direction
                        # Calculate projections of normals onto the current axis direction
                        projection_magnitudes = np.dot(normals, axis)

                        # Calculate orthogonal density by minimizing projections along the axis direction
                        orthogonal_density = np.sum(np.abs(projection_magnitudes))

                        # Slightly adjust the axis direction to maximize orthogonal density
                        rotation_matrix = np.eye(3) + 0.1 * np.random.randn(3, 3)  # Small random rotation
                        rotated_direction = rotation_matrix @ axis
                        rotated_direction /= np.linalg.norm(rotated_direction)

                        new_projection_magnitudes = np.dot(normals, rotated_direction)
                        new_orthogonal_density = np.sum(np.abs(new_projection_magnitudes))

                        # Update if the new direction yields a better orthogonal alignment
                        if new_orthogonal_density < orthogonal_density:
                            axis_direction = rotated_direction
                            orthogonal_density = new_orthogonal_density

                    line_pts = np.array(
                        [axis_point, axis_point + axis * 0.1])  # Extend line in the direction of axis
                    lineset_lines = np.array([[0, 1]], dtype=np.int32)  # Only one line segment

                    print(f"Axis: {axis}")
                    # print(f"Axis Point: {axis_point}")

                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(line_pts)
                    line_set.lines = o3d.utility.Vector2iVector(lineset_lines)

                    # Project all the points onto an orthogonal plane to the axis
                    projection, basis_x, basis_y = mmdl.project_points_to_plane(points, axis_point, axis)

                    # Estimate the circle center and radius
                    cylinder_center, cylinder_radius = mmdl.estimate_circle_center_radius(projection, basis_x, basis_y, axis_point)

                    print(f"Cylinder Center: {cylinder_center}")
                    print(f"Cylinder Radius: {cylinder_radius}")

                    # Create a cylinder object
                    cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=0.15)
                    # cylinder_mesh.paint_uniform_color([0.1, 0.1, 0.7])

                    # Translate the cylinder to the center of the cylinder
                    # translated_mesh = cylinder_mesh.translate(cylinder_center, relative=False)

                    # Rotate the cylinder to align with the axis of the cylinder
                    # cylinder_mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz(axis), center=cylinder_center)

                    # Display everything
                    cylinder_mesh.compute_vertex_normals()
                    o3d.visualization.draw_geometries([pcd, line_set, cylinder_mesh])

                    # # Save inliers to a text file
                    # np.savetxt('all_points.txt', points, fmt='%.8f', delimiter=',')

                print(f"Total points extracted: {len(points)}")
                if len(points) > 0:
                    print(f"Sample points: {points[:5]}")  # Print the first 5 points
                else:
                    print("No valid points extracted.")

                if len(depth_frame) > 0:
                    # Display the isolated color frame
                    cv2.namedWindow("Isolated Color Frame", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("Isolated Color Frame", isolated_color_frame)
                    # Exponentially increase the depth values for better visualization
                    isolated_depth_frame = np.exp(isolated_depth_frame / 400)
                    isolated_depth_frame = cv2.normalize(isolated_depth_frame, None, 0, 200, cv2.NORM_MINMAX)
                    cv2.namedWindow("Isolated Depth Frame", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("Isolated Depth Frame", isolated_depth_frame)

    # Show the original color frame for reference
    # cv2.imshow("Color Frame", color_frame)

    # Update the FPS counter
    fps.update()

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Stop the FPS counter and release the video stream
fps.stop()
rs_cam.stop()
cv2.destroyAllWindows()
