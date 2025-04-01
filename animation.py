import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import mne
from mne import surface
import os
from tvb.simulator.lab import connectivity


def load_brain_surface():
    # Set your actual subjects_dir path where fsaverage is located
    subjects_dir = 'C:/Users/thorb/PycharmProjects/Internship II - Psychiatric Analysis/preprocessed_data'
    subject = 'fsaverage'

    # Load the left and right hemisphere surface meshes
    surf_lh = surface.read_surface(f'{subjects_dir}/{subject}/surf/lh.pial')  # Left hemisphere surface
    surf_rh = surface.read_surface(f'{subjects_dir}/{subject}/surf/rh.pial')  # Right hemisphere surface

    # surf_lh will return a tuple (vertices, faces)
    vertices_lh, faces_lh = surf_lh
    vertices_rh, faces_rh = surf_rh

    return vertices_lh, faces_lh, vertices_rh, faces_rh


def animate_brain_activity_3d_with_overlay(sim_data):
    #animation.writers['ffmpeg'].bin = os.path.join(os.getcwd(), 'animations/ffmpeg.exe')
    print('WRITER',animation.writers.list())  # This will print out the available writers
    vertices_lh, faces_lh, vertices_rh, faces_rh = load_brain_surface()

    time, brain_activity = sim_data[0]
    num_regions = brain_activity.shape[2]
    num_timepoints = len(time)

    conn = connectivity.Connectivity.from_file()
    region_coords = conn.centres[:num_regions]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(vertices_lh[:, 0], vertices_lh[:, 1], vertices_lh[:, 2], triangles=faces_lh, color='gray',
                    alpha=0.3)
    ax.plot_trisurf(vertices_rh[:, 0], vertices_rh[:, 1], vertices_rh[:, 2], triangles=faces_rh, color='gray',
                    alpha=0.3)

    scatter = ax.scatter(region_coords[:, 0], region_coords[:, 1], region_coords[:, 2],
                         c=np.linspace(0, 1, len(region_coords)), s=50, cmap='viridis')

    # Function to update the plot for each frame (i.e., each time step)
    def update(frame):
        activity = brain_activity[frame, 0, :, 0]

        # Avoid division by zero when normalizing
        if np.max(activity) - np.min(activity) > 0:
            norm_activity = (activity - np.min(activity)) / (np.max(activity) - np.min(activity))
        else:
            norm_activity = np.zeros_like(activity)
        print('frame:', frame)

        scatter.set_array(norm_activity)  # Set color based on activity
        scatter.set_cmap('viridis')  # Ensure colormap is applied

        ax.set_title(f"Time: {time[frame]:.2f} ms")
        return scatter,

    print(num_timepoints)
    ani = animation.FuncAnimation(fig, update, frames=10, interval=50, blit=False)

    #save_path = os.path.join('animations', "EEG_10s.mp4")  # Change to .mp4 for video
    #ani.save(save_path, writer='ffmpeg', fps=20)

    plt.show()
