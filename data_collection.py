import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from loguru import logger
import time
from mmwave.dataloader import DCA1000
from mmwave.dataloader.radars import TI
import datetime
from radar_processing import processing

# FPS calculation
prev_time_1 = time.time()
prev_time_2 = time.time()
frame_count_1 = 0
frame_count_2 = 0

# Configuration parameters
#############################################################################################################################
start_freq = 77e9
freq_slope = 100e6 / 1e-6
end_freq = 80.9e9
sampling_rate = 7.2e6
num_sample = 256
num_chirp_loops = 32
idle_time = 7e-6
ramp_end_time = 39e-6
adc_valid_start_time = 3e-6
frame_periodicity = 50e-3
speed_of_light = 3e8
num_TX = 3
num_RX = 4
virtual_ant = num_TX * num_RX
distance = 1.974e-3

maximum_beat_freq = min(10e6, 0.8 * sampling_rate)
chirp_time = num_sample / sampling_rate
valid_sweep_bandwith = chirp_time * freq_slope
chirp_repetition_time = num_TX * (idle_time + ramp_end_time)
carrier_freq = start_freq + freq_slope * adc_valid_start_time + valid_sweep_bandwith / 2

maximum_range = maximum_beat_freq * speed_of_light / (2 * freq_slope)
range_resolution = speed_of_light / (2 * valid_sweep_bandwith)
maximum_velocity = speed_of_light / (4 * chirp_repetition_time * carrier_freq)
velocity_resolution = 2 * maximum_velocity / num_chirp_loops

logger.info(f"maximum_beat_freq: {maximum_beat_freq}")
logger.info(f"chirp_time: {chirp_time}")
logger.info(f"valid_sweep_bandwith: {valid_sweep_bandwith}")
logger.info(f"chirp_repetition_time: {chirp_repetition_time}")
logger.info(f"carrier_freq: {carrier_freq}")
logger.info(f"maximum_range: {maximum_range}")
logger.info(f"range_resolution: {range_resolution}")
logger.info(f"maximum_velocity: {maximum_velocity}")
logger.info(f"velocity_resolution: {velocity_resolution}")
#############################################################################################################################

dca = None
radar = None
stop_flag = False
frameNumInBuf = 1
numframes = 10000

def on_key_event(event):
    global stop_flag
    if event.name == 'esc':
        stop_flag = True

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
decimate.set_option(rs.option.filter_magnitude, 4)

# Threshold filter to remove background
threshold_filter = rs.threshold_filter()
threshold_filter.set_option(rs.option.max_distance, 6)

def get_point_cloud(isRaw=False):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()

    if isRaw:
        # Output the raw point cloud without filtering
        depth_frame_used = depth_frame
    else:
        # Apply the threshold filter and output the filtered point cloud
        depth_frame_filtered = threshold_filter.process(depth_frame)
        depth_frame_used = depth_frame_filtered

    depth_frame_used = decimate.process(depth_frame_used)

    # Calculate the point cloud
    points = pc.calculate(depth_frame_used)
    verts = np.asarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

    return verts


def cartesian_to_polar(x, y, z):
    range_ = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)  # Keep azimuth in radians for polar plot
    return range_, azimuth

if __name__ == '__main__':
    # Set figure and axis font sizes
    font_size = 64  # Define a standard font size for all elements

    fig = plt.figure(figsize=(80, 25), dpi=20)
    ax1 = fig.add_subplot(131, polar=True)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, polar=False)

    # scatter = ax1.scatter([], [], s=1, c='red')
    # scatter3 = ax3.scatter([], [], s=1, c='red')

    dca = DCA1000()
    dca.reset_radar()
    dca.reset_fpga()
    time.sleep(1)
    dca_config_file = "cf.json" 
    radar_config_file = "iwr18xx_profile.cfg"
    radar = TI(cli_loc='COM5', data_loc='COM6',data_baud=921600, config_file=radar_config_file,verbose=True)
    _,_,ADC_PARAMS,_=DCA1000.read_config(radar_config_file)
    radar.setFrameCfg(numframes)
    radar.create_read_process(numframes)
    dca.configure(dca_config_file,radar_config_file)
    logger.debug("press ENTER to start capture...")
    radar.start_read_process()
    dca.stream_start()
    startTime = datetime.datetime.now()
    radar.startSensor()

    while True and not stop_flag:
        # NTP time and FPS for subfigure 1 (mmWave Radar)
        current_time_1 = time.time()
        frame_count_1 += 1
        fps_1 = frame_count_1 / (current_time_1 - prev_time_1)
        ntp_time_1 = datetime.datetime.now().strftime('%H:%M:%S.%f')

        # NTP time and FPS for subfigure 2 (Depth Camera)
        current_time_2 = time.time()
        frame_count_2 += 1
        fps_2 = frame_count_2 / (current_time_2 - prev_time_2)
        ntp_time_2 = datetime.datetime.now().strftime('%H:%M:%S.%f')

        # depth cam
        verts = get_point_cloud()
        if verts.size == 0:
            continue
        x = verts[:, 0]
        y = verts[:, 1]
        z = verts[:, 2]
        range_cam, azimuth_cam = cartesian_to_polar(x, y, z)

        # mm radar
        data_buf = dca.fastRead_in_Cpp(frameNumInBuf, sortInC=True)
        adc_data = np.reshape(data_buf, (-1, ADC_PARAMS['chirps'], ADC_PARAMS['tx'], ADC_PARAMS['rx'], ADC_PARAMS['samples']//2, ADC_PARAMS['IQ'], 2))
        adc_data = np.transpose(adc_data, (0, 1, 2, 3, 4, 6, 5))
        adc_data = np.reshape(adc_data, (-1, ADC_PARAMS['chirps'], ADC_PARAMS['tx'], ADC_PARAMS['rx'], ADC_PARAMS['samples'], ADC_PARAMS['IQ']))
        adc_data = (1j * adc_data[:,:,:,:,:,0] + adc_data[:,:,:,:,:,1]).astype(np.complex64)
        adc_data = adc_data[:, 0, :,:,:]
        processed_data = np.abs(processing(adc_data=adc_data))

        # crop the spectrum to align the depth cam and mmwave radar
        processed_data = processed_data[:, 14:101, :142]

        # Convert azimuth from radians to degrees for the depth camera and adjust to the correct FOV range
        azimuth_cam_deg = np.degrees(azimuth_cam)

        # Adjust azimuth to match the depth camera FOV
        azimuth_cam_deg = (azimuth_cam_deg / max(abs(np.min(azimuth_cam_deg)), abs(np.max(azimuth_cam_deg)))) * (depth_cam_config['fov_azi'] / 2)

        # Filter points based on range
        mask = (range_cam >= 0) & (range_cam <= depth_cam_config['max_range'])
        range_cam = range_cam[mask]
        azimuth_cam = azimuth_cam[mask]
        azimuth_cam_deg = azimuth_cam_deg[mask]
        # pc_cam3 = np.c_[range_cam, azimuth_cam_deg]

        # Update the radar subplot
        ax2.clear()
        ax2.imshow(processed_data[0], extent=[0, depth_cam_config['max_range'], -depth_cam_config['fov_azi'] / 2, depth_cam_config['fov_azi'] / 2], aspect='auto', cmap='viridis')
        ax2.set_xlabel("Range (m)", fontsize=font_size)
        ax2.set_ylabel("Azimuth (degrees)", fontsize=font_size)
        ax2.set_title(f"mmWave Radar Range-Azimuth Spectrum\nNTP Time: {ntp_time_2} | FPS: {fps_2:.2f}", fontsize=font_size)
        ax2.set_xlim(0, depth_cam_config['max_range'])
        ax2.set_ylim(-depth_cam_config['fov_azi'] / 2, depth_cam_config['fov_azi'] / 2)
        ax2.tick_params(axis='both', which='major', labelsize=font_size)

        # Update the depth camera subplot
        ax1.clear()
        ax1.scatter(azimuth_cam, range_cam, s=3, c='red', label='Depth Camera')
        ax1.set_title(f"Depth Camera Range-Azimuth Pointcloud\nNTP Time: {ntp_time_1} | FPS: {fps_1:.2f}", fontsize=font_size)
        ax1.set_xlabel("Range (m)", fontsize=font_size)
        ax1.set_ylabel("Azimuth (degrees)", fontsize=font_size)
        ax1.set_ylim(0, depth_cam_config['max_range'])
        ax1.set_xlim(-depth_cam_config['fov_azi'] / 2 * np.pi / 180, depth_cam_config['fov_azi'] / 2 * np.pi / 180)
        ax1.tick_params(axis='both', which='major', labelsize=font_size)

        # Update the combined subplot
        ax3.clear()
        ax3.imshow(processed_data[0], extent=[0, depth_cam_config['max_range'], -depth_cam_config['fov_azi'] / 2, depth_cam_config['fov_azi'] / 2], aspect='auto', cmap='viridis', alpha=0.5)
        ax3.scatter(range_cam, azimuth_cam_deg, s=3, c='red', label='Depth Camera')
        ax3.set_xlabel("Range (m)", fontsize=font_size)
        ax3.set_ylabel("Azimuth (degrees)", fontsize=font_size)
        ax3.set_title(f"Combined Range-Azimuth Spectrum\nNTP Time: {ntp_time_1} | FPS: {fps_1:.2f}", fontsize=font_size)
        ax3.set_xlim(0, depth_cam_config['max_range'])
        ax3.set_ylim(-depth_cam_config['fov_azi'] / 2, depth_cam_config['fov_azi'] / 2)
        ax3.tick_params(axis='both', which='major', labelsize=font_size)


        # ax3.scatter(range_cam, azimuth_cam, s=1, c='red', label='Depth Camera')
        # scatter3.set_offsets(pc_cam)
        # ax3.set_xlabel("Range (m)", fontsize=font_size)
        # ax3.set_ylabel("Azimuth (degrees)", fontsize=font_size)
        # ax3.set_title(f"Combined Range-Azimuth Spectrum\nNTP Time: {ntp_time_1} | FPS: {fps_1:.2f}", fontsize=font_size)


        # Add boundary lines on the second subplot
        # ax2.axhline(y=depth_cam_config['fov_azi'] / 2, color='red', linewidth=1)
        # ax2.axhline(y=-depth_cam_config['fov_azi'] / 2, color='red', linewidth=1)
        # ax2.axvline(x=depth_cam_config['max_range'], color='red', linewidth=1)
        # logger.info(f"The boundary is: {depth_cam_config['max_range']} m, +- {depth_cam_config['fov_azi'] / 2} deg")
        # get the boundry bins
        # logger.info(f"Range Boundary bins: {int(depth_cam_config['max_range'] / range_resolution)}")    # 142
        # logger.info(f"Azimuth Boundary bins: {int(depth_cam_config['fov_azi'] / 2 / (360 / processed_data.shape[1]))}") # 14, 14+87

        plt.draw()
        plt.pause(0.1)
