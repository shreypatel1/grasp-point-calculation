import pyrealsense2 as rs
import numpy as np
import json


class RealSenseCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        json_file = "config/rs_depth_config2.json"
        with open(json_file, "r") as f:
            depth_config = json.load(f)

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        # device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

        advanced_mode = rs.rs400_advanced_mode(device)
        if not advanced_mode.is_enabled():
            print("Device is not in advanced mode, enabling...")
            advanced_mode.toggle_advanced_mode(True)
        advanced_mode.load_json(json.dumps(depth_config))

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

    def get_color_intrinsics(self):
        return self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def get_depth_intrinsics(self):
        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        return self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics(), depth_scale

    def get_extrinsics(self):
        return self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(self.pipeline.get_active_profile().get_stream(rs.stream.color))
