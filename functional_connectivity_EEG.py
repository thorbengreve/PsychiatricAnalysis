import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import mne  # for EEG preprocessing
from sklearn.decomposition import PCA
import powerlaw

import networkx as nx


# Load EEG data (Example: channels x timepoints)
# Replace this with your own data loading mechanism
def load_eeg_data(file_path):
    # Example: Assume EEG data is stored in a NumPy array (channels x timepoints)
    return np.load(file_path)


# Compute functional connectivity measures
def compute_connectivity(eeg_data, sfreq, method='correlation'):
    """
    Compute functional connectivity of EEG data.

    Parameters:
        eeg_data (numpy array): EEG data of shape (channels, timepoints)
        sfreq (float): Sampling frequency of EEG data
        method (str): Connectivity measure ('correlation', 'coherence', 'plv')

    Returns:
        connectivity_matrix (numpy array): Functional connectivity matrix (channels x channels)
    """
    n_channels = eeg_data.shape[0]

    if method == 'correlation':
        # Compute Pearson correlation between channels
        connectivity_matrix = np.corrcoef(eeg_data)

    elif method == 'coherence':
        # Compute coherence between all pairs of channels
        connectivity_matrix = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                f, Cxy = signal.coherence(eeg_data[i], eeg_data[j], fs=sfreq, nperseg=1024)
                connectivity_matrix[i, j] = np.mean(Cxy)  # Average coherence across frequencies

    elif method == 'plv':
        # Compute phase-locking value (PLV)
        analytic_signal = signal.hilbert(eeg_data, axis=1)
        phase_data = np.angle(analytic_signal)
        connectivity_matrix = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                phase_diff = np.exp(1j * (phase_data[i] - phase_data[j]))
                connectivity_matrix[i, j] = np.abs(np.mean(phase_diff))

    else:
        raise ValueError("Invalid method. Choose 'correlation', 'coherence', or 'plv'.")

    return connectivity_matrix


# Visualize the connectivity matrix
def plot_connectivity(connectivity_matrix, method):
    plt.figure(figsize=(8, 6))
    plt.imshow(connectivity_matrix, cmap='hot', interpolation='nearest')
    plt.title(f'EEG Functional Connectivity ({method})')
    plt.colorbar(label='Connectivity Strength')
    plt.xlabel('Channels')
    plt.ylabel('Channels')
    plt.show()


# Convert the connectivity matrix to a graph
def analyze_graph(connectivity_matrix, threshold=0.5):
    G = nx.Graph()
    num_channels = connectivity_matrix.shape[0]

    # Thresholding to remove weak connections
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            if np.abs(connectivity_matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=connectivity_matrix[i, j])

    # Compute graph metrics
    degree_centrality = nx.degree_centrality(G)
    clustering_coeff = nx.average_clustering(G)
    path_length = nx.average_shortest_path_length(G)

    print(f"Clustering Coefficient: {clustering_coeff}")
    print(f"Average Path Length: {path_length}")
    return G, degree_centrality, clustering_coeff, path_length

# Example usage
if __name__ == "__main__":
    # Simulated EEG data (128 channels, 5000 timepoints)
    eeg_data = np.random.randn(128, 5000)
    sfreq = 256  # Sampling frequency in Hz

    # Compute and plot functional connectivity
    for method in ['correlation', 'coherence', 'plv']:
        connectivity_matrix = compute_connectivity(eeg_data, sfreq, method=method)
        plot_connectivity(connectivity_matrix, method)
