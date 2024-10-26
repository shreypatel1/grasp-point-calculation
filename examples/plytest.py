import open3d as o3d
import numpy as np

# Load the PLY file
ply_file_path = "examples/1.ply"
pcd = o3d.io.read_point_cloud(ply_file_path)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

# Convert the point cloud to a NumPy array
points = np.asarray(pcd.points)
print(points)
