import pyrealsense2 as rs
import numpy as np


class RealSenseCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

    def get_frames(self):
        # Wait for frames from the Realsense Camera
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("No frame received!")
            return False, None, None

        # Convert images to numpy arrays
        depth_frame = np.asanyarray(depth_frame.get_data())
        color_frame = np.asanyarray(color_frame.get_data())

        return True, depth_frame, color_frame

    def stop(self):
        self.pipeline.stop()

    def get_intrinsics(self):
        return self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
