import pyrealsense2 as rs
import numpy as np
import cv2
import json
import open3d as o3d

pc = rs.pointcloud()

pipeline = rs.pipeline()
config = rs.config()

json_file = "rs_depth_config2.json"
with open(json_file, "r") as f:
    depth_config = json.load(f)

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))


config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start streaming
try:
    pipeline.start(config)
except RuntimeError as e:
    print("RuntimeError:", e)

advanced_mode = rs.rs400_advanced_mode(device)
if not advanced_mode.is_enabled():
    print("Device is not in advanced mode, enabling...")
    advanced_mode.toggle_advanced_mode(True)
advanced_mode.load_json(json.dumps(depth_config))

while True:
    # Wait for frames from the Realsense Camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        print("No frame received!")
        break

    points = pc.calculate(depth_frame)
    vertices = np.asanyarray(points.get_vertices())  # Shape: [N, 3] where N is the number of points
    # print(vertices)


    # Create an Open3D point cloud object
    # point_cloud = o3d.geometry.PointCloud()
    # if len(vertices) > 0:
    #     point_cloud.points = o3d.utility.Vector3dVector(vertices)
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window()
    #     vis.add_geometry(point_cloud)
    #     vis.run()
    #     vis.destroy_window()

    # Convert images to numpy arrays
    color_frame = np.asanyarray(color_frame.get_data())
    depth_frame = np.asanyarray(depth_frame.get_data())
    print(f"Depth Frame size: {len(depth_frame[0])} x {len(depth_frame)}")
    print(f"Color Frame size: {len(color_frame[0])} x {len(color_frame)}")

    cv2.imshow("Color Frame", color_frame)
    cv2.imshow("Depth Frame", depth_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break
