import numpy as np
from tvb.simulator.lab import models, connectivity, simulator, monitors
import matplotlib.pyplot as plt
import tvb
from tvb.datatypes.sensors import SensorsEEG
from tvb.simulator.monitors import EEG
import mne
import os

'''

SIMULATIONS with The Virtual Brain

open GUI with: 
python -m tvb.interfaces.web.run

'''


def get_hydrocel_128_leadfield(subject="fsaverage", subjects_dir="preprocessed_data", bem_exists=True):
    """
    Compute or load a lead field matrix for a 128-channel HydroCel EEG cap.

    Parameters:
        subject (str): Name of the subject (default: 'fsaverage' for template brain).
        subjects_dir (str): Path to FreeSurfer/MNE subjects directory.
        bem_exists (bool): If True, assumes a precomputed BEM model exists.

    Returns:
        np.ndarray: Lead field matrix (shape: 128 x N_sources)
    """
    leadfield_path = os.path.join(subjects_dir, subject, "hydrocel_128_leadfield.npy")

    # Load precomputed lead field if available
    if os.path.exists(leadfield_path):
        print(f"Loading precomputed lead field from {leadfield_path}")
        return np.load(leadfield_path)

    print("Computing lead field matrix...")

    # Load standard 128-channel HydroCel EEG montage
    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")

    # Create EEG info object
    info = mne.create_info(montage.ch_names, sfreq=1000, ch_types="eeg")  # Fake EEG setup
    info.set_montage(montage)

    # Compute BEM model (if not already done)
    if not bem_exists:
        bem_model = mne.make_bem_model(subject=subject, subjects_dir=subjects_dir)
        bem_solution = mne.make_bem_solution(bem_model)
    else:
        bem_solution = mne.read_bem_solution(f"{subjects_dir}/{subject}/bem/{subject}-bem-sol.fif")

    # Load or create source space
    src = mne.setup_source_space(subject, spacing="oct6", subjects_dir=subjects_dir)

    # Load or create MRI-EEG transformation
    trans = "fsaverage"

    # Compute forward model
    fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem_solution, eeg=True, meg=False)

    # Extract lead field matrix (gain matrix)
    lead_field = fwd["sol"]["data"]  # Shape: (128 EEG sensors, N brain sources)

    # Save for future use
    np.save(leadfield_path, lead_field)
    print(f"Lead field matrix saved at {leadfield_path}")

    return lead_field


# print(get_hydrocel_128_leadfield())
# print(get_hydrocel_128_leadfield().shape)

def create_tvb_model(coupling_strength, nsig, ntau, noisy, dt=0.1):
    conn = connectivity.Connectivity.from_file()
    conn.weights = conn.weights / np.max(conn.weights)
    conn.configure()
    model = models.JansenRit()
    lead_field = np.load('preprocessed_data/fsaverage/hydrocel_128_leadfield.npy')  # Simulated lead field matrix
    # eeg_monitor = monitors.EEG(sensors=lead_field)

    noise = tvb.simulator.lab.noise.Additive(nsig=np.array([nsig]), ntau=ntau)

    if noisy:
        sim = simulator.Simulator(
            model=model,
            connectivity=conn,
            coupling=simulator.coupling.Linear(a=np.array([coupling_strength])),
            integrator=simulator.integrators.EulerStochastic(dt=dt, noise=noise),
        )
    else:
        sim = simulator.Simulator(
            model=model,
            connectivity=conn,
            coupling=simulator.coupling.Linear(a=np.array([coupling_strength])),
            integrator=simulator.integrators.EulerStochastic(dt=dt),
            # monitors=[eeg_monitor],
            # simulation_length=1000
        )
    # eeg_sensors = SensorsEEG()
    # eeg_sensors.loc = lead_field  # Assign lead field to sensor locations
    # eeg_sensors.labels = np.array([f"Ch{i + 1}" for i in range(lead_field.shape[0])], dtype=str)  # Assign labels
    # eeg_sensors.configure()  # Required in TVB
    # eeg_monitor = EEG(sensors=eeg_sensors)
    #
    # sim.monitors = [eeg_monitor]
    sim.configure()

    return sim


def plot_simulation(sim_data):
    time, brain_activity = sim_data[0]  # Extract time array

    # Plot the activity of the first brain region
    plt.figure(figsize=(10, 5))
    # print('SHAPE:', brain_activity.shape)                               # (1000, 4, 76, 1)
    # print('NUMBER OF REGIONS:', brain_activity.shape[2])
    for region in range(3):  # brain_activity.shape[2]):
        plt.plot(time, brain_activity[:, 0, region], label=f"Region {region} activity")  # Adjust indexing if needed
    plt.xlabel("Time (ms)")
    plt.ylabel("Activity")
    plt.title("Simulated Brain Activity")
    plt.legend()
    plt.show()


def normalize_data(data):
    data = np.array(data)  # Ensure it's a NumPy array
    min_val = np.nanmin(data)  # Get min while ignoring NaNs
    max_val = np.nanmax(data)  # Get max while ignoring NaNs

    if max_val - min_val == 0:  # Avoid division by zero
        return np.zeros_like(data)

    return (data - min_val) / (max_val - min_val)


def export_array(sim_data, normalize):
    data_list = []
    for sub in range(1):
        time, brain_activity = sim_data[0]  # Extract time array
        eeg_signal = brain_activity[:, 0, :, 0]
        if normalize:
            eeg_signal = normalize_data(eeg_signal)
        data_list.append(eeg_signal)
    return data_list
