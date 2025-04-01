import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import entropy
from scipy.signal import welch
from antropy import perm_entropy
from scipy.signal import hann
from scipy.stats import linregress


def get_psd_with_welch(activity_array, sampling_rate, nperseg=256, noverlap=None):
    # If noverlap is not provided, default to 50% overlap
    if noverlap is None:
        noverlap = nperseg // 2

    frequencies, psd_welch_like = welch(activity_array, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
    valid_indices = (frequencies > 0)
    log_freqs = np.log10(frequencies[valid_indices])
    log_powers = np.log10(psd_welch_like[valid_indices])
    slope, intercept, r_value, p_value, std_err = linregress(log_freqs, log_powers)

    return slope, log_freqs, intercept, frequencies, psd_welch_like


def get_exp(activity_array, sampling_rate):
    """ OLD FUNCTION """
    fft_result = np.fft.rfft(activity_array)
    frequencies = np.fft.rfftfreq(len(activity_array), d=1 / sampling_rate)
    power_spectrum = np.abs(fft_result) ** 2
    valid_indices = (frequencies > 0)  # & (frequencies < sampling_rate / 2)  # Exclude 0 Hz and very high freqs
    log_freqs = np.log10(frequencies)[valid_indices]
    log_powers = np.log10(power_spectrum)[valid_indices]
    slope, intercept, r_value, p_value, std_err = linregress(log_freqs, log_powers)
    return slope, log_freqs, intercept, frequencies, power_spectrum


def get_p_value_and_r_squared(exp_MDD, exp_HC):
    # Perform Mann-Whitney U Test
    U, p_value = stats.mannwhitneyu(exp_MDD, exp_HC, alternative='two-sided')

    # Compute Z-score from U
    N1, N2 = len(exp_MDD), len(exp_HC)
    N = N1 + N2
    mean_U = (N1 * N2) / 2
    std_U = np.sqrt((N1 * N2 * (N1 + N2 + 1)) / 12)
    Z = (U - mean_U) / std_U

    # Compute effect size (r and r^2)
    r = Z / np.sqrt(N)
    r_squared = r ** 2

    return p_value, r_squared


def hurst_exponent(array):
    """
    Calculate the Hurst Exponent of a time series using the rescaled range method.
    :param array: Time series data (1D array)
    :return: Estimated Hurst exponent
    """
    N = len(array)
    Y = np.cumsum(array - np.mean(array))  # Remove mean and cumulative sum
    R = np.max(Y) - np.min(Y)  # Range
    S = np.std(array)  # Standard deviation
    return np.log(R / S) / np.log(N)


def box_count(signal, scale):
    reshaped = signal[:(len(signal) // scale) * scale].reshape(-1, scale)
    max_vals = np.max(reshaped, axis=1)
    min_vals = np.min(reshaped, axis=1)
    return np.sum(max_vals - min_vals)


def self_similarity(activity_array):
    scales = [2, 4, 8, 16, 32, 64, 128, 256]
    counts = np.array([box_count(activity_array, s) for s in scales])  # Convert to NumPy array
    log_scales = np.log(scales)
    log_counts = np.log(counts + 1e-10)
    slope, intercept, r_value, p_value, std_err = linregress(log_scales, log_counts)
    fitted_line = slope * log_scales + intercept

    plt.figure()
    plt.plot(log_scales, log_counts, 'bo', label='Log-Log Data')  # Blue points for data
    plt.plot(log_scales, fitted_line, 'r-', label=f'Linear Fit: slope = {slope:.2f}')  # Red line for fit
    plt.plot(np.log(scales), np.log(100 * np.array(scales) ** -1.5), label='Power Law (α = 1.5)', linestyle='--')

    plt.xlabel('Log(Scale)')
    plt.ylabel('Log(Counts)')
    plt.legend()
    plt.title('Log-Log Plot of Box Counting and Linear Fit')
    plt.show()

    fractal_dimension = -slope
    return fractal_dimension


from scipy.linalg import lstsq


def detrended_fluctuation_analysis(signal, min_window=4, max_window=None, num_windows=20):
    """
    Perform DFA on a given 1D signal.

    Parameters:
        signal (np.array): The EEG signal (1D array).
        min_window (int): Minimum window size.
        max_window (int): Maximum window size (defaults to len(signal)//4).
        num_windows (int): Number of window sizes to consider.

    Returns:
        alpha (float): Scaling exponent.
        scales (np.array): Window sizes.
        flucts (np.array): Fluctuation function values.
    """
    N = len(signal)
    if max_window is None:
        max_window = N // 4  # Reasonable max window

    # Integrate signal
    integrated_signal = np.cumsum(signal - np.mean(signal))

    # Logarithmically spaced window sizes
    scales = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), num=num_windows).astype(int))
    flucts = []

    for scale in scales:
        segments = N // scale
        rms_list = []

        for i in range(segments):
            idx_start = i * scale
            idx_end = idx_start + scale
            if idx_end > N:
                break

            segment = integrated_signal[idx_start:idx_end]
            x = np.vstack([np.ones(scale), np.arange(scale)]).T
            (slope, intercept), _, _, _ = lstsq(x, segment)  # Linear fit
            detrended = segment - (slope * np.arange(scale) + intercept)
            rms_list.append(np.sqrt(np.mean(detrended ** 2)))

        flucts.append(np.mean(rms_list))

    # Fit power law (linear fit in log-log space)
    log_scales = np.log10(scales)
    log_flucts = np.log10(flucts)
    alpha, _ = np.polyfit(log_scales, log_flucts, 1)

    # plt.figure(figsize=(6, 4))
    # plt.plot(log_scales, log_flucts, 'o', label="DFA Data")
    # plt.plot(log_scales, alpha * log_scales + _, label=f"Fit (α={alpha:.2f})")
    # plt.xlabel("log(Window size)")
    # plt.ylabel("log(Fluctuation)")
    # plt.legend()
    # plt.show()

    return alpha, scales, flucts


def shannon_entropy(signal):
    """Compute Shannon entropy of a 1D signal."""
    signal = signal[~np.isnan(signal)]
    hist, _ = np.histogram(signal, bins=256, density=True)
    # hist = hist[hist > 0]  # Remove zero probabilities
    return entropy(hist)


def spectral_entropy(signal, fs):
    """Compute spectral entropy of a 1D signal."""
    freqs, psd = welch(signal, fs=fs)
    psd_norm = psd / np.sum(psd)  # Normalize power spectral density
    return entropy(psd_norm)


def compute_permutation_entropy(signal, order=3, delay=1):
    """Compute permutation entropy."""
    return perm_entropy(signal, order=order, delay=delay)


def get_exponent(data_list, sampling_rate):
    print('COMPUTING POWER LAW EXPONENT')
    exp_list = []
    for subject, data in enumerate(data_list):
        exp_per_file = []
        print(subject)
        for channel in range(data.shape[1]):
            exp, _, _, _, _ = get_psd_with_welch(data[:, channel], sampling_rate)
            exp_per_file.append(exp)
        exp_list.append(exp_per_file)
    return exp_list


def get_summed_signal(data_list):
    print('COMPUTING SUMMED SIGNAL')
    sum_list = []
    for subject, data in enumerate(data_list):
        sum_per_file = []
        for channel in range(data.shape[1]):
            sum = np.sum(data[:, channel])  # get
            sum_per_file.append(sum)
        sum_list.append(sum_per_file)
    return sum_list


def get_hurst(data_list):
    print('COMPUTING HURST EXPONENT')
    hurst_list = []
    for subject, data in enumerate(data_list):
        hurst_per_file = []
        for channel in range(data.shape[1]):
            exp = hurst_exponent(data[:, channel])  # get
            hurst_per_file.append(exp)
        hurst_list.append(hurst_per_file)
    return hurst_list


def get_fractaldim(data_list):
    print('COMPUTING FRACTAL DIMENSION')
    fd_list = []
    for subject, data in enumerate(data_list):
        print(subject)
        hurst_per_file = []
        for channel in range(data.shape[1]):
            fractal_dim = self_similarity(data[:, channel])
            hurst_per_file.append(fractal_dim)
        fd_list.append(hurst_per_file)
    return fd_list


def get_DFA(data_list):
    DFA_list = []
    print('COMPUTING DETRENDED FLUCTUATION ANALYSIS')
    for subject, data in enumerate(data_list):
        DFA_per_file = []
        print(subject)
        for channel in range(data.shape[1]):
            DFA, _, _ = detrended_fluctuation_analysis(data[:, channel])
            DFA_per_file.append(DFA)
        DFA_list.append(DFA_per_file)
    return DFA_list


def get_shannon_entropy(data_list):
    print('COMPUTING SHANNON ENTROPY')
    entr_list = []
    for subject, data in enumerate(data_list):
        entr_per_file = []
        for channel in range(data.shape[1]):
            entr = shannon_entropy(data[:, channel])
            entr_per_file.append(entr)
        entr_list.append(entr_per_file)
    return entr_list


def get_spectral_entropy(data_list, fs):
    print('COMPUTING SPECTRAL ENTROPY')
    entr_list = []
    for subject, data in enumerate(data_list):
        entr_per_file = []
        for channel in range(data.shape[1]):
            entr = spectral_entropy(data[:, channel], fs)
            entr_per_file.append(entr)
        entr_list.append(entr_per_file)
    return entr_list


def get_perm_entropy(data_list):
    print('COMPUTING PERMUTATION ENTROPY')
    entr_list = []
    for subject, data in enumerate(data_list):
        entr_per_file = []
        for channel in range(data.shape[1]):
            entr = compute_permutation_entropy(data[:, channel])
            entr_per_file.append(entr)
        entr_list.append(entr_per_file)
    return entr_list


def combine_MDD_HC(MDD, HC):
    list_all = MDD + HC
    return list_all
