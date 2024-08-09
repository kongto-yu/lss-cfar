
import numpy as np
from loguru import logger

def range_fft(data):
    '''
    input: data with shape (frame, slow time, tx, rx, fast time)
    return: range spectrum with shape (frame, slow time, tx, rx, range bins)
    '''
    return np.fft.fft(data,axis=-1)

def form_virtual_antennas(data, isAzimuthOnly=True):
    if isAzimuthOnly:
        # remove the second TX channel
        data = data[:, [0, 2], :, :]
        data = np.reshape(data, (data.shape[0], -1, data.shape[-1]))
        num_virtual_antennas = data.shape[2]
    else:
        data = np.reshape(data, (data.shape[0], -1, data.shape[-1]))
        num_virtual_antennas = data.shape[2]
    return data, num_virtual_antennas

def Bartlett_doa_estimation(data ,azimuth_degree_range=(-60, 60)):
    M = data.shape[1]  # Total antenna elements = TX * RX
    azimuth_angles = np.deg2rad(np.arange(*azimuth_degree_range))
    steering_vectors = np.exp(1j * np.pi * np.outer(np.arange(M), np.sin(azimuth_angles)))
    beamformed_spectrum = np.einsum('fmr,ma->far', data, steering_vectors)
    
    return beamformed_spectrum

def processing(adc_data):
    # logger.info the parameters
    range_spectrum = range_fft(adc_data)
    # logger.info(f"shape of range_spectrum: {range_spectrum.shape}")

    virtual_antennas_data, num_virtual_antennas = form_virtual_antennas(range_spectrum)
    # logger.info(f"shape of virtual_antennas: {virtual_antennas_data.shape}, num_virtual_antennas: {num_virtual_antennas}")

    aoa_spectrogram = Bartlett_doa_estimation(virtual_antennas_data)
    # logger.info(f"shape of aoa_spectrogram: {aoa_spectrogram.shape}")

    return aoa_spectrogram