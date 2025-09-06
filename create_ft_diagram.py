#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

from data_processing import *


def apply_sfft(signal: np.ndarray, nperseg=128, fs=2048):
    f, t, Zxx = stft(
        signal,
        fs=fs,
        nperseg=nperseg,
    )
    magnitude = np.log1p(np.abs(Zxx))

    return f, t, magnitude


def normalize(x):
    m = np.max(np.abs(x))
    if m == 0:
        return np.zeros(len(x))
    return x / m


if __name__ == "__main__":

    zoom = 2

    sd = read_hdf_file()

    if len(sys.argv) > 1:
        sample_id = int(sys.argv[1])
    else:
        sample_id = 0

    strain = sd.h1_strain[sample_id]
    signal = sd.h1_signal[sample_id]

    fig, [[ax2, ax0], [ax3, ax1]] = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    strain_f, strain_t, strain_magnitude = apply_sfft(strain)
    signal_f, signal_t, signal_magnitude = apply_sfft(signal)

    print(strain_magnitude.shape)

    strain_t -= 8
    signal_t -= 8

    ax0.pcolormesh(strain_t, strain_f, strain_magnitude, shading="gouraud")
    ax1.pcolormesh(signal_t, signal_f, signal_magnitude, shading="gouraud")

    ax0.set_ylabel("Frequency (HZ)", fontsize=16)
    ax1.set_ylabel("Frequency (Hz)", fontsize=16)

    ax0.set_title("Spectrogram", fontsize=16)

    xlim_start = -0.5
    xlim_end = 0.1

    ax1.set_xticks(np.arange(xlim_start, xlim_end, 0.1))
    ax1.set_xlim([-0.5, 0.1])

    x = np.linspace(min(strain_t), max(strain_t), len(strain))

    fig.supxlabel("Time from event (s)", fontsize=16)

    ax2.set_title("Signal", fontsize=16)

    ax2.plot(x, normalize(strain))
    ax2.set_yticks([-1, 0, 1])

    ax2.set_ylabel("Whitened Signal Strain", color="C0", fontsize=16)

    ax3.plot(x, normalize(signal), color="C1")
    ax3.set_ylabel("Raw Signal Strain", color="C1", fontsize=16)
    ax3.set_yticks([-1, 0, 1])

    plt.subplots_adjust(wspace=0.8, hspace=.1)
    # plt.tight_layout()

    plt.savefig("img/ft-diagram.png", bbox_inches="tight")
    # plt.show()
