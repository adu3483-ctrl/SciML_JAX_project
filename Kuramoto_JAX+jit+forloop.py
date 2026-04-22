import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
import time

# 1. Setup Parameters
N, K = 1000, 2.0
dt, steps = 0.1, 2000

# 2. NumPy Initialization (As requested)
rng = np.random.default_rng(42)
omega_np = rng.normal(loc=0.0, scale=1.0, size=N)
theta_np = rng.uniform(low=0.0, high=2 * np.pi, size=N)

# Transform into JAX arrays for the GPU/XLA
omega = jnp.array(omega_np)
init_theta = jnp.array(theta_np)

# 3. The JIT-ed Function with a Python Loop
@jit
def run_simulation_jit_loop(theta):
    # In JAX, arrays are immutable, so we use this specific syntax to update indices
    r_history = jnp.zeros(steps)
    
    # This Python loop will be "unrolled" during tracing
    for i in range(steps):
        # Calculate Order Parameter
        z = jnp.mean(jnp.exp(1j * theta))
        r = jnp.abs(z)
        # Store r using the 'at[index].set' syntax (required inside JIT)
        r_history = r_history.at[i].set(r)
        
        # RK4 Math
        def f(th):
            return omega + (K/N) * jnp.sin(th[None, :] - th[:, None]).sum(axis=1)
        
        k1 = f(theta)
        k2 = f(theta + dt/2 * k1)
        k3 = f(theta + dt/2 * k2)
        k4 = f(theta + dt * k3)
        theta = theta + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
    return theta, r_history

# --- EXECUTION BLOCK ---

# WARNING: Compilation might take 30-60 seconds because of loop unrolling!
print("Compiling (Unrolling 2000 steps)...")
warmup_start = time.perf_counter()
_, _ = run_simulation_jit_loop(init_theta)
print(f"Compilation finished in {time.perf_counter() - warmup_start:.2f}s")

print("Starting JIT-ed Loop benchmark...")
start = time.perf_counter()

# 1. Execute
final_theta, r_history = run_simulation_jit_loop(init_theta)

# 2. Block until finished
r_history.block_until_ready()

end = time.perf_counter()
print(f"JAX (JIT-ed Loop) Execution time: {end - start:.4f} seconds")

# Plotting
plt.plot(np.arange(steps) * dt, r_history)
plt.title(f"JAX JIT-ed Python Loop (N={N})")
plt.xlabel("Time $t$")
plt.ylabel("Order Parameter $r$")
plt.show()