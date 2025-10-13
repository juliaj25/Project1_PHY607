

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

# Runge-Kutta 4th order implementation
def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def rk4_solver(f, t_span, y0, dt):
    t_values = np.arange(t_span[0], t_span[1] + dt, dt)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        y_values[i] = rk4_step(f, t_values[i-1], y_values[i-1], dt)
    return t_values, y_values



# Initial conditions
y0 = np.array([1.0, 0.0])  # initial displacement and velocity
t_span = (0, 20)           # time interval
dt = 0.05                  # time step

# Solve using Euler
t_euler, y_euler = euler_solver(damped_oscillator, t_span, y0, dt)


# Solve using RK4
t_rk4, y_rk4 = rk4_solver(damped_oscillator, t_span, y0, dt)

# Solve using SciPy's solve_ivp 
sol = solve_ivp(damped_oscillator, t_span, y0, t_eval=np.arange(t_span[0], t_span[1]+dt, dt))

# Plot results
plt.figure(figsize=(12, 6))

plt.plot(t_euler, y_euler[:, 0], 'r--', label='Euler method')

plt.plot(t_rk4, y_rk4[:, 0], 'b-.', label='RK4 method')
plt.plot(sol.t, sol.y[0], 'k', label='SciPy solve_ivp ')

plt.xlabel('Time [s]')
plt.ylabel('Displacement [x]')
plt.title('Damped Harmonic Oscillator: Displacement over Time')
plt.legend()
plt.grid(True)
plt.show()