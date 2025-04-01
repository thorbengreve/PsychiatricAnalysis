import numpy as np
import mne
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import methods


def plot_electrodes(p_values, label, scale=200):
    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")

    # Extract x, y positions of electrodes
    pos = montage.get_positions()["ch_pos"]
    ch_names = montage.ch_names
    xy = np.array([pos[ch][:2] for ch in ch_names])  # Extract only (x, y)

    cmap_name = "jet_r"  # Try: 'viridis', 'coolwarm', 'inferno', 'magma', 'jet'
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Plot topomap
    fig, ax = plt.subplots(figsize=(6, 6))
    cbar = fig.colorbar(sm, ax=ax, pad=0.05)
    cbar.set_label("P-Value")
    ax.set_title(f"EEG 128 Electrodes - {label} Mapping")
    im, _ = mne.viz.plot_topomap(p_values, xy, axes=ax, cmap=cmap, sensors=True, res=300,
                                 contours=0, outlines='head')  # , size=sizes / 100)

    plt.show()


def plot_electrodes_p_value(exp_list_MDD, exp_list_HC, n_channel, scale=200):
    p_list = []
    for channel in range(n_channel):
        exp_list_MDD_per_channel = []
        for patient in range(len(exp_list_MDD)):
            exp_list_MDD_per_channel.append(exp_list_MDD[patient][channel])
        exp_list_HC_per_channel = []
        for control in range(len(exp_list_HC)):
            exp_list_HC_per_channel.append(exp_list_HC[control][channel])
        p, r2 = methods.get_p_value_and_r_squared(exp_list_MDD_per_channel, exp_list_HC_per_channel)
        p_list.append(p)
    p_array = np.array(p_list)
    print(p_array)
    plot_electrodes(p_array, 'P-value', scale)


def plot_electrodes_2(MDD, HC, label, scale=200):
    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")

    # Extract x, y positions of electrodes
    pos = montage.get_positions()["ch_pos"]
    ch_names = montage.ch_names
    xy = np.array([pos[ch][:2] for ch in ch_names])  # Extract only (x, y)

    cmap_name = "jet_r"  # Try: 'viridis', 'coolwarm', 'inferno', 'magma', 'jet'
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=min(min(MDD), min(HC)), vmax=max(max(MDD), max(HC)))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Plot topomap
    fig, ax = plt.subplots(figsize=(6, 6))
    cbar = fig.colorbar(sm, ax=ax, pad=0.05)
    cbar.set_label(label)
    ax.set_title("MDD")
    im1, _ = mne.viz.plot_topomap(MDD, xy, axes=ax, cmap=cmap, sensors=True, res=300,
                                  contours=0, outlines='head')  # , size=sizes / 100)
    plt.show()


def compute_mean_plot(exp_list_MDD, exp_list_HC, label, n_channel):
    mean_list_MDD = []
    for channel in range(n_channel):
        exp_per_channel_list_MDD = []
        for sub in exp_list_MDD:
            exp = sub[channel]
            exp_per_channel_list_MDD.append(exp)
        mean_per_channel_MDD = np.mean(exp_per_channel_list_MDD)
        mean_list_MDD.append(mean_per_channel_MDD)

    mean_list_HC = []
    for channel in range(n_channel):
        exp_per_channel_list_HC = []
        for sub in exp_list_HC:
            exp = sub[channel]
            exp_per_channel_list_HC.append(exp)
        mean_per_channel_HC = np.mean(exp_per_channel_list_HC)
        mean_list_HC.append(mean_per_channel_HC)
    print(mean_list_MDD)
    print(mean_list_HC)
    plot_electrodes_2(mean_list_MDD, mean_list_HC, label)
