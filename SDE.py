import numpy as np
import matplotlib.pyplot as plt


def simulate_sde(a=1.0, b=1.5, sigma=0.5, dt=0.01, T=100):
    """ Simulate the stochastic differential equation: dx = (-4ax^3 + 2bx) dt + sigma dW """

    N = int(T / dt)  # Number of time steps
    x = np.zeros(N)  # State variable
    x[0] = np.random.uniform(-1, 1)  # Initial condition

    for i in range(1, N):
        dW = np.sqrt(dt) * np.random.randn()  # Wiener process increment
        x[i] = x[i - 1] + (-4 * a * x[i - 1] ** 3 + 2 * b * x[i - 1]) * dt + sigma * dW

    return x, np.linspace(0, T, N)


# Simulate and plot
x, t = simulate_sde()
plt.figure(figsize=(10, 4))
plt.plot(t, x, label='Brain State')
plt.axhline(y=0, color='k', linestyle='--', label='Energy Barrier')
plt.xlabel("Time")
plt.ylabel("State (Mood/Neural Activity)")
plt.title("Stochastic Brain State Transitions")
plt.legend()
plt.show()
