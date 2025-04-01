import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation


# Let's assume you have brain region positions (e.g., from a template brain surface)
# For this example, let's create random spherical coordinates for brain regions.
def generate_random_coordinates(num_regions):
    # Generate random points on the unit sphere
    theta = np.random.uniform(0, 2 * np.pi, num_regions)
    phi = np.random.uniform(0, np.pi, num_regions)

    # Convert spherical coordinates to Cartesian coordinates (x, y, z)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.vstack((x, y, z)).T


# Simulated activity data (random for this example, replace with your actual simulation)
def generate_simulated_activity(num_regions, num_timepoints):
    return np.random.randn(num_timepoints, num_regions)


# Main 3D animation function
def animate_brain_activity_3d(sim_data):
    time, brain_activity = sim_data[0]
    num_regions = brain_activity.shape[2]
    num_timepoints = len(time)

    # Generate random coordinates for the brain regions
    region_coords = generate_random_coordinates(num_regions)

    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of brain regions
    scatter = ax.scatter(region_coords[:, 0], region_coords[:, 1], region_coords[:, 2], c='k', s=50)

    # Function to update the plot for each frame (i.e., each time step)
    def update(frame):
        # Get the activity of the regions at the current time point
        activity = brain_activity[frame, 0, :]

        # Normalize the activity to use as color intensity
        norm_activity = (activity - np.min(activity)) / (np.max(activity) - np.min(activity))

        # Update the color of each region based on its activity
        scatter.set_facecolor(plt.cm.viridis(norm_activity))  # Use a colormap (e.g., viridis)

        ax.set_title(f"Time: {time[frame]:.2f} ms")
        return scatter,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_timepoints, interval=50, blit=False)

    # Show the animation
    plt.show()


# Example usage
sim_data = run_simulation(create_tvb_model())  # Assuming you have your simulation data
animate_brain_activity_3d(sim_data)
