import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, jit, lax
import matplotlib.pyplot as plt
import time

# 1. Setup Parameters
N, K = 10000, 2.0
dt, steps = 0.1, 2000
key = random.PRNGKey(42)
k1, k2 = random.split(key)


omega = random.normal(k1, (N,))
init_theta = random.uniform(k2, (N,), minval=0, maxval=2*jnp.pi)

# 2. Define the "Scan-friendly" RK4 Step
def simulation_step(theta, _):
    """
    theta: the 'carry' (current phases)
    _: the 'x' (a dummy input since we don't need external values)
    """
    # Our derivative function (Kuramoto)
    def f(th): 
        return omega + (K/N) * jnp.sin(th[None, :] - th[:, None]).sum(axis=1)
    
    # Standard RK4 math
    k1 = f(theta)
    k2 = f(theta + dt/2 * k1)
    k3 = f(theta + dt/2 * k2)
    k4 = f(theta + dt * k3)
    next_theta = theta + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Calculate order parameter r to "log" it for history
    # r = |1/N * sum(e^(i*theta))|
    z = jnp.mean(jnp.exp(1j * theta))
    r = jnp.abs(z)
    
    # Return (state_for_next_step, value_to_save_in_history)
    return next_theta, r

# 3. Compile and Run the WHOLE simulation at once
# We wrap the scan in a JIT to optimize the entire 2000-step sequence
@jit
def run_full_simulation(initial_theta):
    # lax.scan(function, initial_state, sequence_of_steps)
    final_theta, r_history = lax.scan(simulation_step, initial_theta, jnp.arange(steps))
    return final_theta, r_history

# NO! We should NOT run the simulation here to "warm up" the JIT, 
# because it will execute the entire 2000-step simulation during 
# the warm-up, this ended up jamming the hardware and delayed the 
# execution of final_theta, r_history = run_full_simulation(init_theta).
# # Warm up (Compiles here)
# _, _ = run_full_simulation(init_theta)

# --- Corrected Execution Block ---
start = time.perf_counter()

# 1. Get the results (this returns immediately due to async dispatch)
final_theta, r_history = run_full_simulation(init_theta)

# 2. Block until the math is actually finished
r_history.block_until_ready() 

end = time.perf_counter()

print(f"Execution time: {end - start:.4f} seconds")
# Convert iterations to time by multiplying by dt (0.1)
plt.plot(jnp.arange(steps) * dt, r_history, label='JAX Order Parameter $r(t)$')
plt.xlabel("Time $t$")
plt.ylabel("Order Parameter $r$")
plt.title("Synchronization over Time")
plt.ylim(0, 1.05)
plt.show()