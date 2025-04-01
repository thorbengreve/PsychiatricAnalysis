import scipy.io
from sklearn.decomposition import PCA
from methods import get_psd_with_welch, hurst_exponent
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.stats import linregress
import mne
import methods
from scipy.stats import gaussian_kde
import matplotlib.animation as animation


def plot_signal_and_FFT(data_list, sampling_rate, channel_wise=True,
                        only_one=False):  # shape: [(timepoints, channels),...]
    for k, data in enumerate(data_list):
        if only_one:
            if k == 1:
                return
        if not channel_wise:
            figure, axes = plt.subplots(1, 2, figsize=(15, 5))
        if len(data.shape) < 2:
            data = np.reshape(data, (-1, 1))
        for channel in range(data.shape[1]):
            if channel_wise:
                figure, axes = plt.subplots(1, 2, figsize=(15, 5))
            signal = data[:, channel]
            axes[0].plot([y * (1 / sampling_rate) for y in range(len(signal))], signal)
            # axes[0].set_title('EEG-Signal')
            axes[0].set_title('Actigraph Metric')
            axes[0].set_xlabel('Time [s]')
            axes[0].set_ylabel('Normalized activity')

            slope, log_freqs, intercept, positive_freqs, positive_magnitudes = get_psd_with_welch(signal,sampling_rate)
            fitted_line = slope * log_freqs + intercept

            #axes[1].plot(positive_freqs[2:], positive_magnitudes[2:], markersize=4, label=f'Channel {channel+1}', alpha=0.7)  # 'o'
            axes[1].plot(positive_freqs[2:], positive_magnitudes[2:], markersize=4, label='Data', alpha=0.7)  # 'o'
            axes[1].plot(10 ** log_freqs, 10 ** fitted_line, color='red', label=f'Fit: slope={slope:.4f}')

            axes[1].set_xscale('log')
            axes[1].set_yscale('log')
            axes[1].set_title('PSD')
            axes[1].set_xlabel('Frequency [Hz]')
            axes[1].set_ylabel('Power')
            axes[1].grid(which="both", ls="--", linewidth=0.5)
            axes[1].legend()
            if channel_wise:
                plt.tight_layout()
                plt.show()
        if not channel_wise:
            plt.tight_layout()
            plt.show()


def plot_exponent_hist(exp_list_MDD, exp_list_HC, n_channel, label, density=True):
    if len(exp_list_MDD) > 1:
        exp_list_MDD_flat = np.concatenate(exp_list_MDD)
        exp_list_MDD_flat = exp_list_MDD_flat.tolist()
    else:
        exp_list_MDD_flat = exp_list_MDD[0]
    if len(exp_list_HC) > 1:
        exp_list_HC_flat = np.concatenate(exp_list_HC)
        exp_list_HC_flat = exp_list_HC_flat.tolist()
    else:
        exp_list_HC_flat = exp_list_HC[0]

    exp_list_MDD_flat = np.array(exp_list_MDD_flat)[~np.isnan(exp_list_MDD_flat)]
    exp_list_HC_flat = np.array(exp_list_HC_flat)[~np.isnan(exp_list_HC_flat)]
    bins = 30
    bin_range = (min(min(exp_list_MDD_flat), min(exp_list_HC_flat)), max(max(exp_list_MDD_flat), max(exp_list_HC_flat)))

    p, r2 = methods.get_p_value_and_r_squared(exp_list_MDD_flat, exp_list_HC_flat)
    plt.figure()
    print('LENGTH: ', len(exp_list_MDD_flat))

    plt.hist(exp_list_MDD_flat, bins, range=bin_range, density=density, alpha=0.5)
    plt.hist(exp_list_HC_flat, bins, range=bin_range, density=density, alpha=0.4)

    # DENSITY FUNCTION
    kde_MDD = gaussian_kde(exp_list_MDD_flat)
    x = np.linspace(min(exp_list_MDD_flat), max(exp_list_MDD_flat), 1000)  # Range for the curve
    kde_curve_MDD = kde_MDD(x)
    plt.plot(x, kde_curve_MDD, color='blue', label='Smoothed Density (KDE)')
    kde_HC = gaussian_kde(exp_list_HC_flat)
    x = np.linspace(min(exp_list_HC_flat), max(exp_list_HC_flat), 1000)  # Range for the curve
    kde_curve_HC = kde_HC(x)
    plt.plot(x, kde_curve_HC, color='orange', label='Smoothed Density (KDE)')

    # plt.axvline(np.mean(exp_list_MDD_flat), color='blue')
    # plt.axvline(np.mean(exp_list_HC_flat), color='orange')
    # plt.legend(['MDD', 'HC'])
    print('p:',p, 'r^2:', r2)
    plt.title(label)
    plt.show()

    for channel in range(n_channel):
        exp_list_MDD_per_channel = []
        for patient in range(len(exp_list_MDD)):
            exp_list_MDD_per_channel.append(exp_list_MDD[patient][channel])
        exp_list_HC_per_channel = []
        for control in range(len(exp_list_HC)):
            exp_list_HC_per_channel.append(exp_list_HC[control][channel])
        p, r2 = methods.get_p_value_and_r_squared(exp_list_MDD_per_channel, exp_list_HC_per_channel)
        if p < 0.05:
            print('Channel=', channel, ' p=', p, ' r^2=', r2)
            plt.figure()
            # plt.axvline(np.mean(exp_list_MDD_per_channel), color='blue')
            # plt.axvline(np.mean(exp_list_HC_per_channel), color='orange')
            plt.hist(exp_list_MDD_per_channel, bins, range=bin_range, alpha=0.5, density=density)
            plt.hist(exp_list_HC_per_channel, bins, range=bin_range, alpha=0.4, density=density)
            plt.legend(['MDD', 'HC'])
            plt.title(f'{label} per Channel ' + str(channel) + ', p=' + str(np.round(p, 4)))
            plt.show()


def plot_exp_against_metric(PHQ9, exp_list, label):
    plt.figure()
    plt.title(f'PHQ9 against {label}')
    plt.xlabel('PHQ9')
    plt.ylabel(label)

    mean_exp = [np.mean(exp_list[sub]) for sub in range(len(exp_list))]
    plt.scatter(PHQ9, mean_exp, label='data')
    slope, intercept, r, p, se = linregress(PHQ9, mean_exp)
    print('SLOPE: ', slope)
    print('r^2: ', r**2)
    print('p: ', p)
    x_regression = np.linspace(0, 27, 100)  # More points to make the line smooth
    y_regression = slope * x_regression + intercept
    formatted_string = f'slope={slope:.4f}, $r^2$={r ** 2:.4f}, p={p:.4f}'
    plt.plot(x_regression, y_regression, color='red', label=formatted_string, linewidth=2)
    plt.legend()
    plt.show()
