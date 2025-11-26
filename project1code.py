# Import packages

import os
import numpy as np
import math as math
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid, simpson


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

plt.plot(t_rk4, y_rk4[:, 0], color='black',  linestyle='-.',
         linewidth=2.5, label='RK4 method')
plt.plot(sol.t, sol.y[0], color='orange', label='SciPy solve_ivp ')

plt.xlabel('Time [s]')
plt.ylabel('Displacement [x]')
plt.title('Damped Harmonic Oscillator: Displacement over Time')
plt.legend()
plt.grid(True)
plt.show()

v = y_rk4[:,1] #get velocity
work_integrand = -c * v**2  # F_d * dx = (-c*v)*dx/dt dt = -c*v^2 dt



#Riemann method
def riemann_sum(t, f):
    total = 0.0
    for i in range(len(f) - 1):
        dt = t[i+1] - t[i]
        total += f[i] * dt
    return total

# Trapezoidal Rule
def trapezoidal_rule(t, f):
    total = 0.0
    for i in range(len(f) - 1):
        dt = t[i+1] - t[i]
        total += 0.5 * (f[i] + f[i+1]) * dt
    return total

 #Simpson's Rule
def simpsons_rule(y, t):
    n = len(y) - 1
    if n % 2 == 1:
        n -= 1
        y = y[:n+1]
        t = t[:n+1]
    h = t[1] - t[0]   # uniform spacing
    s = y[0] + y[-1] + 4*np.sum(y[1:n:2]) + 2*np.sum(y[2:n-1:2])
    return s*h/3

# Integrals
W_riemann = riemann_sum(t_rk4, work_integrand)
W_trap = trapezoidal_rule(t_rk4, work_integrand)
W_simp = simpsons_rule(work_integrand, t_rk4)
# SciPy equivalents
W_trap_scipy = trapezoid(work_integrand, t_rk4)
W_simp_scipy = simpson(work_integrand, t_rk4)


print(f"Riemann sum: {W_riemann:.6f}")
print(f"Trapezoidal: {W_trap:.6f}")
print(f"Simpson's:   {W_simp:.6f}")
print(f"SciPy trapezoidal: {W_trap_scipy:.6f}")
print(f"SciPy Simpson's:   {W_simp_scipy:.6f}")

#Checking work-energy theorem
x = y_rk4[:,0]

K = 0.5 * m * v**2          
U = 0.5 * k * x**2          
E = K + U                   

delta_E = E[-1] - E[0]
print(f"ΔE = {delta_E:.6f}")
