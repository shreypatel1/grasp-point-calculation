import cv2
import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# Declare pointcloud object and RealSense pipeline
pc = rs.pointcloud()
points = rs.points()
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to enable depth stream
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start streaming from RealSense camera
pipeline.start(config)

# Open3D visualizer setup
vis = o3d.visualization.Visualizer()
vis.create_window()

# Create a point cloud object for Open3D
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

try:
    while True:
        # Wait for the next set of frames from the camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            print("No depth frame received")
            continue

        # Generate the point cloud from the depth frame
        points = pc.calculate(depth_frame)

        # Get vertices as a NumPy array of shape (N, 3) -> [x, y, z]
        vertices = np.asanyarray(points.get_vertices())

        # Check if we received valid points
        if vertices.size == 0:
            print("No points in the point cloud")
            continue

        # Convert to a proper format for Open3D point cloud
        verts = np.reshape(vertices, (-1, 3))

        # Assign the points to the Open3D point cloud
        pcd.points = o3d.utility.Vector3dVector(verts)

        # Update the Open3D visualizer
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Display the depth image using OpenCV
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("Depth Frame", depth_colormap)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline and close Open3D window
    pipeline.stop()
    vis.destroy_window()
    cv2.destroyAllWindows()
