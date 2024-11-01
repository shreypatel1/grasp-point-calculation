import numpy as np
from scipy.optimize import minimize

class CameraIntrinsics:

    def __init__(self, fx, fy, cx, cy, coeffs):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.coeffs = coeffs


    def get_camera_matrix(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float32)

    def get_distortion_coeffs(self):
        k1, k2, p1, p2, k3 = self.coeffs
        return np.array([k1, k2, p1, p2, k3], dtype=np.float32)


class MathModels:

    def are_points_within_distance(self, points, max_distance):
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if np.linalg.norm(points[i] - points[j]) > max_distance:
                    return False
        return True

    def generate_plane_samples(self, points, num_samples, max_distance):
        samples = []
        pts_idx = np.arange(len(points))
        while len(samples) < num_samples:
            sample_indices = np.random.choice(pts_idx, 3, replace=False)
            sample_points = points[sample_indices]
            if self.are_points_within_distance(sample_points, max_distance):
                samples.append(sample_points)
        return np.array(samples)

    def project_points_to_plane(self, points, plane_point, plane_normal):
        plane_normal = plane_normal / np.linalg.norm(plane_normal)  # Ensure the normal is normalized
        projections = points - plane_point  # Vector from plane_point to each point
        distances = np.dot(projections, plane_normal)[:, np.newaxis]  # Distance along the normal
        projected_points = points - distances * plane_normal  # Projected points on the plane

        # Choose an orthogonal basis on the plane to convert to 2D
        basis_x = np.cross(plane_normal, [1, 0, 0])
        if np.linalg.norm(basis_x) < 1e-6:
            basis_x = np.cross(plane_normal, [0, 1, 0])
        basis_x /= np.linalg.norm(basis_x)
        basis_y = np.cross(plane_normal, basis_x)
        basis_y /= np.linalg.norm(basis_y)

        # Convert to 2D
        projection_2d = np.column_stack((np.dot(projected_points - plane_point, basis_x),
                                    np.dot(projected_points - plane_point, basis_y)))

        return projection_2d, basis_x, basis_y

    def estimate_circle_center_radius(self, points_2d, basis_x, basis_y, plane_point):

        def objective(center):
            distances = np.linalg.norm(points_2d - center, axis=1)
            return np.var(distances)  # Return the variance of distances

        # Initial guess for the center is the mean of the points
        initial_center = np.mean(points_2d, axis=0)
        result = minimize(objective, initial_center)
        best_center = result.x

        # Calculate the average radius from the optimal center
        distances = np.linalg.norm(points_2d - best_center, axis=1)
        radius = np.mean(distances)

        # Convert the 2D center vector back to 3D space
        best_center = best_center[0] * basis_x + best_center[1] * basis_y + plane_point

        return best_center, radius