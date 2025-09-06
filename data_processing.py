from dataclasses import dataclass
from typing import cast

import h5py
import numpy as np
from h5py import Dataset, Datatype, File, Group


@dataclass
class Signal_Data:
    h1_strain: np.ndarray
    l1_strain: np.ndarray

    h1_signal: np.ndarray
    l1_signal: np.ndarray
    injection_snr: np.ndarray


def inspect_hdf_file(file: str = "data/train_better.hdf"):

    def inspect_hdf_value(value: Group | Dataset | Datatype, depth: int = 0):
        def printw(*args):
            print(" " * depth * 4, *args)

        match type(value):
            case h5py.File:

                f = cast(File, value)

                printw("File", f.name)

                for k in f.keys():
                    inspect_hdf_value(f[k], depth + 1)

            case h5py.Group:

                g = cast(Group, value)

                printw(f"Group: {value.name} ({len(g)})")

                for k in g.keys():
                    inspect_hdf_value(g[k], depth + 1)

            case h5py.Dataset:
                printw("Dataset:", value.name)

                d = cast(Dataset, value)

                printw("Shape:", d.shape)

            case h5py.Datatype:
                printw("Datatype", value.name)

    with h5py.File(file) as f:
        inspect_hdf_value(f)


def read_hdf_file(file: str = "data/train_better.hdf") -> Signal_Data:
    with h5py.File(file) as f:
        injection_samples: Group = cast(Group, f["injection_samples"])
        injection_parameters: Group = cast(Group, f["injection_parameters"])

        h1_strain_dtset = cast(Dataset, injection_samples["h1_strain"])
        h1_strain = np.array(h1_strain_dtset)

        l1_strain_dtset = cast(Dataset, injection_samples["l1_strain"])
        l1_strain = np.array(l1_strain_dtset)

        h1_signal_dtset = cast(Dataset, injection_parameters["h1_signal"])
        h1_signal = np.array(h1_signal_dtset)

        l1_signal_dtset = cast(Dataset, injection_parameters["l1_signal"])
        l1_signal = np.array(l1_signal_dtset)

        injection_snr_dtset = cast(Dataset, injection_parameters["injection_snr"])
        injection_snr = np.array(injection_snr_dtset)

    sd = Signal_Data(
        h1_strain=h1_strain,
        l1_strain=l1_strain,
        h1_signal=h1_signal,
        l1_signal=l1_signal,
        injection_snr=injection_snr,
    )

    return sd
