import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, jit, lax, vmap
import matplotlib.pyplot as plt
import time

# 1. Setup Parameters
N = 500
dt, steps = 0.1, 2000
key = random.PRNGKey(42)
k1, k2 = random.split(key)

omega = random.normal(k1, (N,))
init_theta = random.uniform(k2, (N,), minval=0, maxval=2*jnp.pi)

# 2. Define a function that takes K as an argument
def run_single_experiment(K, initial_theta, omega):
    
    def simulation_step(theta, _):
        # f depends on the K passed into run_single_experiment
        def f(th): 
            return omega + (K/N) * jnp.sin(th[None, :] - th[:, None]).sum(axis=1)
        
        k1 = f(theta)
        k2 = f(theta + dt/2 * k1)
        k3 = f(theta + dt/2 * k2)
        k4 = f(theta + dt * k3)
        next_theta = theta + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        r = jnp.abs(jnp.mean(jnp.exp(1j * theta)))
        return next_theta, r

    # lax.scan handles the time loop
    _, r_history = lax.scan(simulation_step, initial_theta, jnp.arange(steps))
    return r_history

# 3. Create a VECTORIZED version of the whole experiment
# We map over the first argument (K), while keeping theta and omega constant (None)
vectorized_sim = vmap(run_single_experiment, in_axes=(0, None, None))

# 4. Define our range of K values
Ks = jnp.array([1.0, 1.5, 2.0, 3.0, 5.0]) # 5 different experiments

# JIT compile the entire vectorized batch
@jit
def run_batch(Ks_input, theta_input, omega_input):
    return vectorized_sim(Ks_input, theta_input, omega_input)

# --- THE "BURDEN" TEST ---

# WARM UP (Compile the vectorized kernel)
_ = run_batch(Ks, init_theta, omega)

print(f"Starting Vectorized Batch of {len(Ks)} experiments...")
start = time.perf_counter()

# Execute all 5 experiments at once
batch_r_history = run_batch(Ks, init_theta, omega)
batch_r_history.block_until_ready()

end = time.perf_counter()
total_time = end - start
print(f"Total Wall Clock Time for batch: {total_time:.4f} seconds")
print(f"Average time per experiment in batch: {total_time/len(Ks):.4f} seconds")

# 5. Plotting results
plt.figure(figsize=(10, 6))
time_axis = jnp.arange(steps) * dt
for i, k_val in enumerate(Ks):
    plt.plot(time_axis, batch_r_history[i], label=f'K = {k_val}')

plt.xlabel("Time $t$")
plt.ylabel("Order Parameter $r$")
plt.title(f"Parallel Kuramoto Experiments (N={N})")
plt.legend()
plt.show()