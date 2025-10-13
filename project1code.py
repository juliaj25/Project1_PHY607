

# Import packages

import os
import numpy as np
import math as math
from matplotlib import pyplot as plt

from scipy.integrate import solve_ivp

# Parameters
m = 1.0     # mass
c = 0.1     # damping coefficient
k = 1.0     # spring constant

# Define the system of ODEs
def damped_oscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = -(c/m)*v - (k/m)*x
    return np.array([dxdt, dvdt])

# Euler method implementation
def euler_step(f, t, y, dt):
    return y + dt * f(t, y)

def euler_solver(f, t_span, y0, dt):
    t_values = np.arange(t_span[0], t_span[1] + dt, dt)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        y_values[i] = euler_step(f, t_values[i-1], y_values[i-1], dt)
    return t_values, y_values


# Initial conditions
y0 = np.array([1.0, 0.0])  # initial displacement and velocity
t_span = (0, 20)           # time interval
dt = 0.05                  # time step

# Solve using Euler
t_euler, y_euler = euler_solver(damped_oscillator, t_span, y0, dt)

# Plot results
plt.figure(figsize=(12, 6))

plt.plot(t_euler, y_euler[:, 0], 'r--', label='Euler method')


plt.xlabel('Time [s]')
plt.ylabel('Displacement [x]')
plt.title('Damped Harmonic Oscillator: Displacement over Time')
plt.legend()
plt.grid(True)
plt.show()