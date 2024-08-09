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

if __name__ == '__main__':
    keyboard.on_release(on_key_event)
    fig, ax = plt.subplots(figsize=(12, 8))

    try:
        dca = DCA1000()
        # 1. reset radar and dca1000
        dca.reset_radar()
        dca.reset_fpga()
        print("wait for reset")
        time.sleep(1)
        # 2. initial para
        dca_config_file = "cf.json" # lvdsMode=2
        radar_config_file = "iwr18xx_profile.cfg" # lvdsStreamCfg -1 0 1 0 (the third bit is 1), adcbufCfg -1 0 1 1 1
        # change the port number
        radar = TI(cli_loc='COM5', data_loc='COM6',data_baud=921600, config_file=radar_config_file,verbose=True)
        # stop after numframes
        radar.setFrameCfg(numframes)

        # 3. read inner chip dsp data from serial 
        radar.create_read_process(numframes)

        dca.configure(dca_config_file,radar_config_file)  # send FPGA command via Ethernet port

        # press enter to start capture
        logger.debug("press ENTER to start capture...")

        # 6. start serial reading
        radar.start_read_process()

        # 7. start via Ethernet port (DCA1000)
        dca.stream_start()

        # 8. start UDP receiving thread
        # numframes_out,sortInC_out = dca.fastRead_in_Cpp_async_start(frameNumInBuf,sortInC=True) # method 1: async

        # 9. start Radar via serial 
        startTime = datetime.datetime.now()
        
        radar.startSensor()
        start = time.time()
        
        _,_,ADC_PARAMS,_=DCA1000.read_config(radar_config_file)

        while not stop_flag:
            data_buf = dca.fastRead_in_Cpp(frameNumInBuf, sortInC=True)
            adc_data = np.reshape(data_buf, (-1, ADC_PARAMS['chirps'], ADC_PARAMS['tx'], ADC_PARAMS['rx'], ADC_PARAMS['samples']//2, ADC_PARAMS['IQ'], 2))
            adc_data = np.transpose(adc_data, (0, 1, 2, 3, 4, 6, 5))
            adc_data = np.reshape(adc_data, (-1, ADC_PARAMS['chirps'], ADC_PARAMS['tx'], ADC_PARAMS['rx'], ADC_PARAMS['samples'], ADC_PARAMS['IQ']))
            adc_data = (1j * adc_data[:,:,:,:,:,0] + adc_data[:,:,:,:,:,1]).astype(np.complex64)
            adc_data = adc_data[:, 0, :,:,:]
            logger.info("Frame: {}".format(adc_data.shape))     
            processed_data = np.abs(processing(adc_data=adc_data))
            logger.info(f"AOA spectrum shape: {processed_data.shape}")

            # Extract azimuth and range data
            azimuths = np.linspace(-60, 60, processed_data.shape[1])
            ranges = np.linspace(0, maximum_range, processed_data.shape[2])

            # Update plot
            ax.clear()
            ax.set_xlabel("Range (m)")
            ax.set_ylabel("Azimuth (degrees)")
            ax.set_title("Range-Azimuth Radar Map")
            cax = ax.imshow(processed_data[0], extent=[0, maximum_range, -60, 60], aspect='auto', cmap='viridis')
            # fig.colorbar(cax, ax=ax, orientation='vertical', label='Intensity')

            plt.draw()
            plt.pause(0.01)

    except Exception as e:
        traceback.print_exc()
    finally:
        if dca is not None:
            dca.close()
        if radar is not None:
            radar.cli_port.close()
