import numpy as np

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
