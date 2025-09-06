#!/usr/bin/env python3

import copy
from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt

from data_processing import *

target_samples = 2048
event_sample = 0


def normalize(x):
    m = np.max(np.abs(x))
    if m == 0:
        return np.zeros(len(x))
    return x / m


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype="band", output="sos")
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def snr_preprocess(
    sd: Signal_Data,
    snr_lower_bound=15,
) -> Signal_Data:
    filter_arr = list(map(lambda x: x >= snr_lower_bound, sd.injection_snr))
    for signal in fields(sd):
        s = getattr(sd, signal.name)
        if signal.name != "injection_snr":
            setattr(sd, signal.name, normalize(s[filter_arr]))
        else:
            setattr(sd, signal.name, s[filter_arr])
    return sd


def bandpass_filter(
    sd: Signal_Data,
    lowcut_hz=20,
    highcut_hz=400,
    fs=2048,
) -> Signal_Data:
    time_start = 14_452
    time_end = 16_500

    global event_sample
    event_sample = 16_384 - time_start

    assert (
        time_end - time_start == target_samples
    ), f"Time series not {target_samples} in length"

    for i in range(len(sd.h1_strain)):
        sd.h1_strain[i] = butter_bandpass_filter(
            sd.h1_strain[i], lowcut_hz, highcut_hz, fs
        )
        # sd.l1_strain[i] = butter_bandpass_filter(
        #     sd.l1_strain[i], lowcut_hz, highcut_hz, fs
        # )

    return sd


def cut_domain(
    sd: Signal_Data,
    time_start: int = 14_452,
    time_end=16_500,
) -> Signal_Data:
    assert time_end - time_start == 2048, "Time series not 2048 in length"

    for signal in fields(sd):
        s = getattr(sd, signal.name)
        if "signal" in signal.name or "strain" in signal.name:
            setattr(sd, signal.name, s[:, time_start:time_end])

    return sd


def wavelet_filter(
    strain: np.ndarray,
    threshold: float,
    maxlevel: int,
) -> np.ndarray:
    wp = pywt.WaveletPacket(strain, wavelet="db4", mode="symmetric", maxlevel=maxlevel)

    new_wp = pywt.WaveletPacket(data=None, wavelet="db4", mode="symmetric")

    for node in wp.get_level(wp.maxlevel, "freq"):
        new_data = pywt.threshold(node.data, threshold, mode="soft")
        new_wp[node.path] = new_data

    denoised_strain = new_wp.reconstruct(update=False)

    return denoised_strain[: len(strain)]


def wavelet_filter_data(
    sd: Signal_Data,
    threshold: float = 0.25,
    maxlevel: int = 5,
) -> Signal_Data:
    for i in range(len(sd.h1_strain)):
        sd.h1_strain[i] = wavelet_filter(sd.h1_strain[i], threshold, maxlevel)
        # sd.l1_strain[i] = wavelet_filter(sd.l1_strain[i], threshold, maxlevel)
    return sd


def mse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum((normalize(x) - normalize(y)) ** 2) / len(y))


def mse_data(sd: Signal_Data) -> float:
    m = len(sd.h1_strain)
    acc = 0.0
    for i in range(m):
        acc += mse(sd.h1_strain[i], sd.h1_signal[i])
    return acc / m


if __name__ == "__main__":

    print("Reading hdf file... ", end="")
    sd = read_hdf_file()
    print("Done!")

    print("Filtering data for snr above 15... ", end="")
    sd = snr_preprocess(sd)
    print("Done!")

    sd_bandpassed = copy.deepcopy(sd)
    print("Applying band pass filter (20hz, 400hz)... ", end="")
    sd_bandpassed = bandpass_filter(sd_bandpassed)
    print("Done!")

    print(f"Cutting data time domain ({target_samples} samples)... ", end="")
    cut_domain(sd)
    cut_domain(sd_bandpassed)
    print("Done!")

    thresholds = np.arange(0, 1, 0.01)
    levels = np.arange(2, 8, 1)

    mses_by_level = {}
    for level in levels:
        mses_by_level[level] = []

    sd_wf = wavelet_filter_data(sd_bandpassed, 0.37, 5)

    sample_ids = [0, 9, 15]


    x = np.linspace(0, 1, target_samples) - event_sample / target_samples

    xlim = [-0.9, 0.05]
    ylim = [-1.3, 1.3]
    yticks = [-1, 0, 1]

    fig1, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(20, 8))

    for i in range(3):
        sample_id = sample_ids[i]

        strain = normalize(sd.h1_strain[sample_id])
        bandpassed_strain = normalize(sd_bandpassed.h1_strain[sample_id])
        denoised_strain = normalize(sd_wf.h1_strain[sample_id])
        raw_signal = normalize(sd_wf.h1_signal[sample_id])
        snr = sd_wf.injection_snr[sample_id]

        ax[0][i].plot(x, strain, color="C0")
        if i == 0:
            ax[0][i].set_ylabel("Whitened Signal Strain", color="C0", fontsize=14)
        ax[0][i].set_xlim(xlim)
        ax[0][i].set_ylim(ylim)
        ax[0][i].set_yticks(yticks)
                
        ax[1][i].plot(x, denoised_strain, color="C2")
        if i == 0:
            ax[1][i].set_ylabel("WPT Filter Strain", color="C2", fontsize=14)
        ax[1][i].set_xlim(xlim)
        ax[1][i].set_ylim(ylim)
        ax[1][i].set_yticks(yticks)
                
        ax[2][i].plot(x, raw_signal, color="C3")
        if i == 0:
            ax[2][i].set_ylabel("Raw Signal Strain", color="C3", fontsize=14)
        ax[2][i].set_xlim(xlim)
        ax[2][i].set_ylim(ylim)
        ax[2][i].set_yticks(yticks)
                
                
        ax[0][i].axvline(x=0, color="black", ls="--", lw=1)
        ax[1][i].axvline(x=0, color="black", ls="--", lw=1)
        ax[2][i].axvline(x=0, color="black", ls="--", lw=1)

        ax[0][i].set_title(f"Sample #{sample_id} (snr: {snr})", fontsize=14)

    fig1.supxlabel("Time from event time (in seconds)", fontsize=16)

    # plt.gcf().set_size_inches(11, 8.2, forward=True)
    # plt.tight_layout(rect=(0.0, 0.0, 0.0, 0.9))
    plt.subplots_adjust(wspace=.2, hspace=0)

    fig1.savefig("img/denoised.png", bbox_inches="tight")

    # plt.show()
