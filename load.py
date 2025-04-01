import numpy as np
import scipy
import os
from scipy.signal import iirnotch, filtfilt
import mne
import matplotlib.pyplot as plt
import pandas as pd


mne.set_log_level("ERROR")

sampling_rates = {"MODMA_3_rs": 250,
                  "MODMA_128_rs": 250,
                  "OpenNeuro_66_rs": 500,
                  "MathTest": 500,
                  "Depresjon": 1 / 60}


def MODMA_3_rs(preprocessing=True, normalize=True):
    print('IMPORT: MODMA 3 Channel EEG resting state')
    directory_path = "data/MODMA/EEG_3channels_resting_lanzhou_2015/"
    data_list_MDD = []
    data_list_HC = []
    exclusion_list_old = ['02010011_still.txt',
                      '02010022_still.txt',
                      '02010025_still.txt',
                      '02010035_still.txt',
                      '02020022_still.txt',  # only channel 1 artifact
                      '02030001_still.txt',
                      '02030008_still.txt',
                      '02030009_still.txt',  # only channel 1 artifact
                      ]
    exclusion_list = []

    for file in os.listdir(directory_path):
        if file.endswith('txt'):
            if file in exclusion_list:
                pass
            else:
                data = np.loadtxt(os.path.join(directory_path, file), delimiter=' ', dtype=float)
                if preprocessing:
                    fs = sampling_rates['MODMA_3_rs']
                    f0 = 50
                    Q = 30  # (higher = narrower notch)
                    b, a = iirnotch(f0, Q, fs)
                    for channel in range(data.shape[1]):
                        data[:, channel] = filtfilt(b, a, data[:, channel], axis=0)

                if normalize:
                    data -= np.mean(data)
                    data /= np.std(data)
                if file.startswith('0201'):
                    data_list_MDD.append(data)
                else:
                    data_list_HC.append(data)

    return data_list_MDD, data_list_HC


def MODMA_128_rs(preprocessing=True, normalize=True):
    print('IMPORT: MODMA 128 Channel EEG resting state')
    directory_path = "data/MODMA/EEG_128channels_resting_lanzhou_2015"
    data_list_MDD = []
    data_list_HC = []
    exclusion_list = []

    for file in os.listdir(directory_path):
        if file.endswith('mat'):
            if file in exclusion_list:
                pass
            else:
                data = scipy.io.loadmat(os.path.join(directory_path, file))
                key_list = list(data.keys())
                data = data[key_list[3]][:128, :]
                data = np.transpose(data)
                if preprocessing:
                    fs = sampling_rates['MODMA_3_rs']
                    f0 = 50
                    Q = 30  # (higher = narrower notch)
                    b, a = iirnotch(f0, Q, fs)
                    for channel in range(data.shape[1]):
                        data[:, channel] = filtfilt(b, a, data[:, channel], axis=0)
                if normalize:
                    data -= np.mean(data)
                    data /= np.std(data)
                if file.startswith('0201'):
                    data_list_MDD.append(data)
                else:
                    data_list_HC.append(data)

    return data_list_MDD, data_list_HC


def OpenNeuro_66_rs(preprocessing=True, normalize=True):
    print('IMPORT: OpenNeuro 66 Channel EEG resting state')
    directory_path = "data/OpenNeuro_rsEEG/"
    data_list_MDD = []
    data_list_HC = []
    exclusion_list = []

    for subject in os.listdir(directory_path):
        subject_dir = os.path.join(directory_path, subject, 'eeg')
        if os.path.isdir(subject_dir):  # Check if it's a directory
            set_file_path = os.path.join(subject_dir, f"{subject}_task-Rest_run-01_eeg.set")
            eeg_data = mne.io.read_raw_eeglab(set_file_path, preload=True)

            # print(eeg_data.info)
            # eeg_data.plot()  # Plot the data again
            # plt.show()
            has_boundaries = any(annot['description'] == 'boundary' for annot in eeg_data.annotations)

            # Check for boundaries and handle accordingly
            if has_boundaries:
                print(subject)
            else:
                eeg_data.set_eeg_reference('average')
                # eeg_data.filter(l_freq=None, h_freq=100.0)

                eeg_data = eeg_data.get_data().T  # Get the raw EEG signals, shape = (channels, timepoints)
                eeg_data = eeg_data[:200000, :]
                print(eeg_data.shape)
                # data = data.T

                if normalize:
                    eeg_data -= np.mean(eeg_data)
                    eeg_data /= np.std(eeg_data)
                data_list_MDD.append(eeg_data)

    print(f"Processed EEG data for {len(data_list_MDD)} participants.")

    return data_list_MDD, data_list_HC


def Depresjon(normalize=True):
    print('IMPORT: Depresjon Actigraph')
    file_list_MDD = [f'data/depresjon/data/condition/condition_{n}.csv' for n in range(1, 24)]
    file_list_HC = [f'data/depresjon/data/control/control_{n}.csv' for n in range(1, 33)]
    data_list_MDD = []
    data_list_HC = []
    exclusion_list = []
    # exclusion_list = ['condition_2.csv',
    #                   'condition_13.csv',
    #                   'condition_16.csv',
    #                   'condition_20.csv',
    #                   'condition_23.csv']

    for file in file_list_MDD:
        if file in exclusion_list:
            pass
        else:
            data = np.genfromtxt(file, delimiter=',', skip_header=1)
            data = data[:, 2]
            data = np.reshape(data, (-1, 1))  # get only activity values
            if normalize:
                data -= np.mean(data)
                data /= np.std(data)
            data_list_MDD.append(data)
    for file in file_list_HC:
        if file in exclusion_list:
            pass
        else:
            data = np.genfromtxt(file, delimiter=',', skip_header=1)
            data = data[:, 2]
            data = np.reshape(data, (-1, 1))
            if normalize:
                data -= np.mean(data)
                data /= np.std(data)
            data_list_HC.append(data)

    return data_list_MDD, data_list_HC


def MathTest(normalize=True):
    print('IMPORT: Theoretical Test')
    data_list_MDD = []
    data_list_HC = []
    fs = 1000  # Sampling frequency (Hz) – this can be defined as needed
    n_sub = 5
    duration = 60  # Duration of the signal (seconds)
    t = np.linspace(0, duration, fs * duration, endpoint=False)  # Time vector
    n_samples = fs * duration
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)[1:]  # Exclude DC component (0 Hz)

    for virt_sub in range(n_sub):
        alpha = 1.5
        # Calculate the amplitude spectrum based on the PSD exponent
        amplitudes = 1 / (freqs ** (alpha / 2))  # The amplitude spectrum is the square root of the PSD
        phases = np.random.uniform(0, 2 * np.pi, size=len(freqs))  # Random phase for each frequency component
        signal = np.zeros_like(t)
        for freq, amp, phase in zip(freqs, amplitudes, phases):
            signal += amp * np.sin(2 * np.pi * freq * t + phase)
        signal = signal[:, np.newaxis]
        data_list_MDD.append(signal)

    for virt_sub in range(n_sub):
        alpha = 2
        amplitudes = 1 / (freqs ** (alpha / 2))
        phases = np.random.uniform(0, 2 * np.pi, size=len(freqs))
        signal = np.zeros_like(t)
        for freq, amp, phase in zip(freqs, amplitudes, phases):
            signal += amp * np.sin(2 * np.pi * freq * t + phase)
        signal = signal[:, np.newaxis]

        if normalize:
            signal -= np.mean(signal)
            signal /= np.std(signal)
        data_list_HC.append(signal)

    return data_list_MDD, data_list_HC

def get_MADRS():
    df = pd.read_csv('data/depresjon/data/scores.csv')
    MADRS = df['madrs1'].fillna(0).to_numpy()
    print(MADRS)
    return MADRS

def get_PHQ_9(dataset):
    print('IMPORT: MODMA PHQ-9')
    if dataset == 'MODMA_3_rs':
        directory_path = "data/MODMA/EEG_3channels_resting_lanzhou_2015/subjects_information_EEG_3channels_resting_lanzhou_2015.xlsx"
    elif dataset == 'MODMA_128_rs':
        directory_path = "data/MODMA/EEG_128channels_resting_lanzhou_2015/subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx"
    else:
        print('SELECT OTHER DATASET FOR PHQ-9)')
        return
    data_list_MDD = []
    data_list_HC = []
    df = pd.read_excel(directory_path)
    column_f_values = df.iloc[:, 5].values  # Column F is the 6th column (0-indexed)
    PHQ9 = np.array(column_f_values)
    print(PHQ9)
    return PHQ9


def get_BDI():
    print('IMPORT: OpenNeuro 66 BDI')
    directory_path = "data/OpenNeuro_rsEEG/participants.tsv"
    data = np.loadtxt(directory_path, delimiter="\t", skiprows=1, usecols=4)  # Assuming there's a header
    no_data = [x for x in range(76, 122)]
    data = np.delete(data, no_data)
    indices_to_remove = [2, 13, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 44,
                         47, 50, 53, 55, 56, 59, 62, 64, 65, 66, 67, 68, 69, 71, 75]
    data = np.delete(data, indices_to_remove)
    return data


def get_depression_metric(dataset):
    if dataset == 'MODMA_3_rs':
        metric = get_PHQ_9('MODMA_3_rs')
    if dataset == 'MODMA_128_rs':
        metric = get_PHQ_9('MODMA_128_rs')

    if dataset == 'Depresjon':
        metric = get_MADRS()
    if dataset == 'OpenNeuro_66_rs':
        metric = get_BDI()
    return metric