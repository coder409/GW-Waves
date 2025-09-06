#!/usr/bin/env python3

import copy
from dataclasses import fields
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

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
    print("Inspecting hdf file:")
    inspect_hdf_file()
    print()

    print("Reading hdf file... ", end="")
    sd = read_hdf_file()
    print("Done!")

    print("h1_strain shape:", sd.h1_strain.shape)
    print("l1_strain shape:", sd.l1_strain.shape)

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

    print("Applying wavelet packet filter...")
    for threshold, level in tqdm(
        product(thresholds, levels), total=len(levels) * len(thresholds), ncols=80
    ):
        sd_wf = wavelet_filter_data(
            copy.deepcopy(sd_bandpassed), float(threshold), int(level)
        )
        mses_by_level[level].append(mse_data(sd_wf))

    best_mse = float("inf")
    best_level = 0
    best_threshold = 0

    for level in levels:
        mses = mses_by_level[level]
        best_mse_idx = np.argmin(mses)
        if mses[best_mse_idx] < best_mse:
            best_mse = mses[best_mse_idx]
            best_threshold = thresholds[best_mse_idx]
            best_level = int(level)

    print("Best level:", best_level)
    print("Best threshold:", best_threshold)
    print("Best MSE:", best_mse)

    sd_wf = wavelet_filter_data(
        copy.deepcopy(sd_bandpassed), best_threshold, best_level
    )

    sample_id = 0

    strain = normalize(sd.h1_strain[sample_id])
    bandpassed_strain = normalize(sd_bandpassed.h1_strain[sample_id])
    denoised_strain = normalize(sd_wf.h1_strain[sample_id])
    raw_signal = normalize(sd_wf.h1_signal[sample_id])
    snr = sd_wf.injection_snr[sample_id]

    x = np.linspace(0, 1, target_samples) - event_sample / target_samples

    xlim = [-0.9, 0.05]
    ylim = [-1.3, 1.3]
    yticks = [-1, 0, 1]

    fig0, [ax0, ax1] = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    fig0.suptitle(f"Sample {sample_id} (snr: {snr})", fontsize=18)

    ax0.plot(x, strain, color="C0")
    ax0.set_ylabel("Whitened Signal Strain", color="C0", fontsize=14)
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)
    ax0.set_yticks(yticks)

    ax1.plot(x, bandpassed_strain, color="C1")
    ax1.set_ylabel("Bandpassed Signal Strain", color="C1", fontsize=14)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_yticks(yticks)

    ax1.set_xlabel("Time from event time (in seconds)", fontsize=16)

    ax0.axvline(x=0, color="black", ls="--", lw=1)
    ax1.axvline(x=0, color="black", ls="--", lw=1)

    fig0.savefig("img/bandpassed.png", bbox_inches="tight")

    fig1, [ax0, ax1, ax2] = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    ax0.plot(x, strain, color="C0")
    ax0.set_ylabel("Whitened Signal Strain", color="C0", fontsize=18)
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)
    ax0.set_yticks(yticks)

    ax1.plot(x, denoised_strain, color="C2")
    ax1.set_ylabel("WPT Filter Strain", color="C2", fontsize=16)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_yticks(yticks)

    ax2.plot(x, raw_signal, color="C3")
    ax2.set_ylabel("Raw Signal Strain", color="C3", fontsize=16)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_yticks(yticks)

    ax2.set_xlabel("Time from event time (in seconds)", fontsize=16)

    ax0.axvline(x=0, color="black", ls="--", lw=1)
    ax1.axvline(x=0, color="black", ls="--", lw=1)
    ax2.axvline(x=0, color="black", ls="--", lw=1)

    fig1.suptitle(f"Sample #{sample_id} (snr: {snr})", fontsize=20)

    plt.gcf().set_size_inches(11, 8.2, forward=True)
    plt.tight_layout(rect=(0.0, 0.0, 0.0, 0.9))
    plt.subplots_adjust(wspace=0, hspace=0)

    fig1.savefig("img/denoised.png", bbox_inches="tight")

    mse_v_thresh_fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for level in levels:
        ax.plot(thresholds, mses_by_level[level], linewidth=2, label=f"level={level}")
    ax.set_title("Mean Squared Error vs. Filter Threshold", fontsize=18)
    ax.set_xlabel("Filter Threshold", fontsize=14)
    ax.set_ylabel("Mean Squared Error", fontsize=14)
    ax.legend()

    mse_v_thresh_fig.savefig("img/mse-v-thresh.png", bbox_inches="tight")

    # plt.show()
