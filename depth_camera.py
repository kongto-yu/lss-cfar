import math
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from loguru import logger

# Define depth camera configurations
depth_cam_config = {
    'd435': {
        'max_range': 3,
        'fov_azi': 69,
        'fov_ele': 42
    },
    'd455': {
        'max_range': 6,
        'fov_azi': 87,
        'fov_ele': 58
    }
}

# Select the camera type
type = 'd455'
depth_cam_config = depth_cam_config[type]
logger.info(f"Using {type} camera")

class AppState:
    def __init__(self):
        self.paused = False
        self.decimate = 0

state = AppState()

# Configure streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

def get_point_cloud():
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()

    depth_frame = decimate.process(depth_frame)

    # Calculate the point cloud
    points = pc.calculate(depth_frame)
    verts = np.asarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

    return verts

def cartesian_to_polar(x, y, z):
    range_ = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)  # Keep azimuth in radians for polar plot
    return range_, azimuth

# Start streaming
try:
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    scatter = ax.scatter([], [], s=1, c='black')

    while True:
        verts = get_point_cloud()

        if verts.size == 0:
            continue

        # Extract X, Y, Z coordinates
        x = verts[:, 0]
        y = verts[:, 1]
        z = verts[:, 2]

        # Convert to polar coordinates
        range_, azimuth = cartesian_to_polar(x, y, z)

        # Filter points based on range
        mask = (range_ >= 0) & (range_ <= depth_cam_config['max_range'])
        range_ = range_[mask]
        azimuth = azimuth[mask]

        scatter.set_offsets(np.c_[azimuth, range_])
        ax.set_title("Range-Azimuth Radar Plot")
        ax.set_ylim(0, depth_cam_config['max_range'])
        ax.set_xlim(-depth_cam_config['fov_azi'] / 2 * np.pi / 180, depth_cam_config['fov_azi'] / 2 * np.pi / 180)  # Convert degrees to radians for azimuth limits

        plt.pause(0.1)

finally:
    pipeline.stop()
    plt.show()
