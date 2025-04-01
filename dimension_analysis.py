import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
import pywt
from numpy.fft import fft
from hurst import compute_Hc
import nolds


def hurst_exponent(ts):
    """Calculate the Hurst exponent using Rescaled Range Analysis."""
    H, c, data = compute_Hc(ts, kind='change', simplified=True)
    fractal_dim = 2 - H
    return H, fractal_dim


def detrended_fluctuation_analysis(ts):
    """Compute the DFA to estimate fractal dimension."""
    n = len(ts)
    scales = np.floor(np.logspace(0.5, np.log10(n / 2), num=20)).astype(int)
    flucts = []

    for s in scales:
        segments = len(ts) // s
        reshaped = ts[:segments * s].reshape(segments, s)
        local_trend = detrend(reshaped, axis=1)
        variance = np.mean(local_trend ** 2, axis=1)
        flucts.append(np.sqrt(np.mean(variance)))

    coeffs = np.polyfit(np.log(scales), np.log(flucts), 1)
    hurst_exp = coeffs[0]
    fractal_dim = 2 - hurst_exp
    return hurst_exp, fractal_dim


def box_counting(ts, n_scales=10):
    """Estimate fractal dimension using the Box-Counting method."""
    dfa = nolds.dfa(ts)
    return dfa


def wavelet_wtmm(ts):
    """Estimate fractal dimension using Wavelet Transform Modulus Maxima method."""
    wavelet = 'cmor'  # Use a continuous wavelet instead of 'haar'
    scales = np.arange(1, len(ts) // 2)

    coeffs, _ = pywt.cwt(ts, scales=scales, wavelet=wavelet)
    max_coeffs = np.max(np.abs(coeffs), axis=1)

    valid_indices = max_coeffs > 0  # Ensure no log of zero
    coeffs_log = np.log(max_coeffs[valid_indices])
    scales_log = np.log(scales[valid_indices])

    slope, _ = np.polyfit(scales_log, coeffs_log, 1)
    fractal_dim = -slope
    return fractal_dim


def fourier_spectrum(ts):
    """Estimate fractal dimension using Fourier power spectrum analysis."""
    freqs = np.fft.fftfreq(len(ts))
    power_spectrum = np.abs(fft(ts)) ** 2
    mask = freqs > 0

    coeffs = np.polyfit(np.log(freqs[mask]), np.log(power_spectrum[mask]), 1)
    fractal_dim = -coeffs[0] / 2
    return fractal_dim


def autocorrelation(ts):
    from statsmodels.tsa.stattools import acf, pacf

    acf_values = acf(ts, nlags=50)
    plt.plot(acf_values)
    plt.show()


def dimensional_analysis(ts):
    H, fractal_dim_hurst = hurst_exponent(ts)
    print(f"Hurst Exponent: {H:.4f}, Estimated Fractal Dimension: {fractal_dim_hurst:.4f}")

    # hurst_dfa, fractal_dim_dfa = detrended_fluctuation_analysis(ts)
    # print(f"DFA Hurst Exponent: {hurst_dfa:.4f}, Estimated Fractal Dimension: {fractal_dim_dfa:.4f}")

    # fractal_dim_box = box_counting(ts)
    # print(f"Box-Counting Estimated Fractal Dimension: {fractal_dim_box:.4f}")

    # fractal_dim_wtmm = wavelet_wtmm(ts)
    # print(f"WTMM Estimated Fractal Dimension: {fractal_dim_wtmm:.4f}")

    # fractal_dim_fourier = fourier_spectrum(ts)
    # print(f"Fourier Spectrum Estimated Fractal Dimension: {fractal_dim_fourier:.4f}")

    # autocorrelation(ts)
    return fractal_dim_hurst
