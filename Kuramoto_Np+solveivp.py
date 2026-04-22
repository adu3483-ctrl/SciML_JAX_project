import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import time

N = 1000 # Number of oscillators
K = 2.0 # Coupling strength


# Initialize the generator
rng = np.random.default_rng()

# Generate N natural frequency samples (omegas) 
# with mean=0 and std_dev=1
omega = rng.normal(loc=0, scale=1, size=N)

# Generate N chaotic initial phases (1D) between 0 and 2π
theta = rng.uniform(low=0, high=2 * np.pi, size=N)

# Reshape to (N, 1) for broadcasting
# K/N * sum(from j=1 to N) sin(θ_j - θ_i), here s would be theta
F = lambda t, s: omega + (K / N) * np.sin(s[None, :] - s[:, None]).sum(axis=1)
# Adjust the time interval length based on observations
t_end = 200
t_eval = np.arange(0, t_end, 0.1)

start = time.perf_counter()

sol_theta = solve_ivp(F, [0, t_end], theta, t_eval=t_eval)

# sol_theta.y has shape (N, len(t_eval)), 
# where each row corresponds to the phase of an oscillator over time
# 1/N * sum(from j=1 to N) e^(i*θ_j)
Z = np.exp(1j * sol_theta.y).mean(axis=0) 
# Z has shape (len(t_eval),) where each element is the order parameter at a given time

r = np.abs(Z) # Magnitude of the order parameter
psi = np.angle(Z) # Phase(angle) of the order parameter

end = time.perf_counter()
print(f"Execution time: {end - start:.4f} seconds")

# Assuming 't' is your time array and 'r_history' is your calculated magnitude
plt.plot(t_eval, r, label='Order Parameter $r(t)$')
plt.xlabel("Time $t$")
plt.ylabel("Order Parameter $r$")
plt.title("Synchronization over Time")
plt.ylim(0, 1.05)
plt.show()
