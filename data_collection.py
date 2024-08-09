import math
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from loguru import logger
import traceback
import time
from mmwave.dataloader import DCA1000
from mmwave.dataloader.radars import TI
import numpy as np
import matplotlib.pylab as plt
import datetime
import keyboard
from loguru import logger
from radar_processing import processing

# Configuration parameters
#############################################################################################################################
start_freq = 77e9   # start frequency (Hz)
freq_slope = 100e6 / 1e-6  # frequency slope (Hz/s)
end_freq = 80.9e9   # end frequency (Hz)
sampling_rate = 7.2e6  # sampling rate (Hz)
num_sample = 256    # number of samples aka fast time
num_chirp_loops = 32    # number of chirp loops aka slow time
idle_time = 7e-6    # idle time between chirps (s)
ramp_end_time = 39e-6   # ramp end time (s)
adc_valid_start_time = 3e-6  # ADC valid start time (s)
frame_periodicity = 50e-3  # frame periodicity (s)
speed_of_light = 3e8    # speed of light (m/s)
num_TX = 3    # number of transmitters
num_RX = 4    # number of receivers
virtual_ant = num_TX * num_RX    # number of virtual antennas
distance = 1.974e-3    # distance between two adjacent antennas (m)

maximum_beat_freq = min(10e6, 0.8 * sampling_rate)  # maximum beat frequency (Hz), for iwr1843boost, the maximum beat frequency is 10MHz
chirp_time = num_sample / sampling_rate  # chirp time (s), fast time / sampling rate
valid_sweep_bandwith = chirp_time * freq_slope  # valid sweep bandwidth (Hz)
chirp_repetition_time = num_TX * (idle_time + ramp_end_time)    # chirp repetition time (s)
carrier_freq = start_freq + freq_slope * adc_valid_start_time + valid_sweep_bandwith / 2  # carrier frequency (Hz)

maximum_range = maximum_beat_freq * speed_of_light / (2 * freq_slope)  # maximum range (m)
range_resolution = speed_of_light / (2 * valid_sweep_bandwith)  # range resolution (m)

maximum_velocity = speed_of_light / (4 * chirp_repetition_time * carrier_freq)  # maximum velocity (m/s)
velocity_resolution = 2 * maximum_velocity / num_chirp_loops  # velocity resolution (m/s)

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

if __name__ == '__main__':
    fig = plt.figure(figsize=(20, 10), dpi=100)
    ax1 = fig.add_subplot(121, polar=True)
    ax2 = fig.add_subplot(122)

    scatter = ax1.scatter([], [], s=1, c='red')
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
        logger.info("Frame: {}".format(adc_data.shape))     
        processed_data = np.abs(processing(adc_data=adc_data))
        # logger.info(f"AOA spectrum shape: {processed_data.shape}")

        # Extract azimuth and range data
        azimuth_radar = np.linspace(-60, 60, processed_data.shape[1])
        ranges_radar = np.linspace(0, maximum_range, processed_data.shape[2])

        ax2.set_xlabel("Range (m)")
        ax2.set_ylabel("Azimuth (degrees)")
        ax2.set_title("Depth Camera Range-Azimuth Ponitcloud")
        cax = ax2.imshow(processed_data[0], extent=[0, maximum_range, -60, 60], aspect='auto', cmap='viridis')

        # Filter points based on range
        mask = (range_cam >= 0) & (range_cam <= depth_cam_config['max_range'])
        range_cam = range_cam[mask]
        azimuth_cam = azimuth_cam[mask]

        scatter.set_offsets(np.c_[azimuth_cam, range_cam])
        ax1.set_title("mmWave Radar Range-Azimuth Spectrum")
        ax1.set_xlabel("Range (m)")
        ax1.set_ylabel("Azimuth (degrees)")
        ax1.set_ylim(0, depth_cam_config['max_range'])
        ax1.set_xlim(-depth_cam_config['fov_azi'] / 2 * np.pi / 180, depth_cam_config['fov_azi'] / 2 * np.pi / 180)  # Convert degrees to radians for azimuth limits

        # Add boundary lines on the second subplot
        ax2.axhline(y=depth_cam_config['fov_azi'] / 2, color='red', linewidth=1)
        ax2.axhline(y=-depth_cam_config['fov_azi'] / 2, color='red', linewidth=1)
        ax2.axvline(x=depth_cam_config['max_range'], color='red', linewidth=1)

        plt.draw()
        plt.pause(0.1)