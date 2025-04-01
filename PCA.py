import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def PCA_channel(eeg_data):
    n_comp = 50
    pca = PCA(n_comp)
    generalized_signal = pca.fit_transform(eeg_data)
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, n_comp + 1), pca.explained_variance_ratio_ * 100, alpha=0.7)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.title("PCA Explained Variance")
    plt.show()
    return generalized_signal


from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler


def ICA_channel(eeg_data):
    n_comp = 128

    # Apply whitening to the data (StandardScaler)
    eeg_data_whitened = StandardScaler().fit_transform(eeg_data)

    # Apply ICA
    ica = FastICA(n_comp, random_state=42)
    independent_components = ica.fit_transform(eeg_data_whitened)

    # Denormalize by scaling the components back
    # Use the mean and std of the original data
    eeg_data_mean = np.mean(eeg_data, axis=0)
    eeg_data_std = np.std(eeg_data, axis=0)

    # Denormalize each component
    denormalized_components = independent_components * eeg_data_std

    # Compute the power of each denormalized component
    component_power = np.var(denormalized_components, axis=0)

    # Plot ICA Component Power
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, n_comp + 1), component_power, alpha=0.7)
    plt.xlabel("Independent Component")
    plt.ylabel("Component Power (Variance)")
    plt.title("ICA Component Power Distribution")
    plt.show()

    return independent_components
