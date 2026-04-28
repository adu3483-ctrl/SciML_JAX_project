---
marp: true
theme: gaia
paginate: true
backgroundColor: #fff
footer: "Numerical Computation with JAX | SciML Presentation 2026"
math: katex
style: |
  /* 針對語法高亮中的註解類別進行調整 */
  section pre code .hljs-comment, 
  section pre code .token.comment {
    color: #555555; /* 你可以根據需求調整這個灰色數值 */
    font-style: italic;        /* 保持斜體（可選） */
  }
  section {
    font-size: 26px;
    font-family: 'Inter', 'Noto Sans TC', sans-serif;
  }
  h1 { color: #2c3e50; font-size: 50px; }
  h2 { color: #34495e; }
  
  /* for code blocks */
  section pre {
    background: #f1f5f9; 
    border-radius: 8px;   /*rounded corner*/
  }

  section pre code {
    color: #0366d6;
  }
  /* for in line codes*/
  section code {
    background: #f1f5f9; /* 跟大區塊一樣的淺藍灰色 */
    color: #0366d6;      /* 你偏好的藍色 */
    padding: 0.1em 0.3em; /* 增加一點點留白，比較不擠 */
    border-radius: 4px;   /* 圓角 */
  }
  blockquote {
    background: #f9f9f9;
    border-left: 10px solid #ccc;
    margin: 1.5em 10px;
    padding: 0.5em 10px;
  }
---
# Introduction to SciML
## From Numerical Foundations to Physics-Informed Learning

**Presenters:**
Liang Chin, Liao Pin Wei (ODE Group)
Chen Yuan Lung (PDE Group)

National Tsing Hua University

---

# The Narrative Arc: A Story in Five Acts

1. **Solve ODEs**: The foundational numerical engine.
2. **Scale with JAX**: Overcoming the $O(N^2)$ Kuramoto bottleneck.
3. **Extend to PDEs**: The Method of Lines bridge.
4. **Differentiate**: Using `jax.grad` for parameter optimization.
5. **Learn Physics**: Transitioning from solvers to **PINNs**.

> *How do we go from a textbook equation to a high-performance, differentiable simulation?*

---

# I. Why Scientific Machine Learning?

- **The Bridge**: Science writes laws as PDEs; computers solve them via discretization.
- **The Wall**: Classical methods hit limits in high dimensions or inverse problems.
- **The SciML Solution**: 
  - Fusing **Domain Knowledge** (Physics) with **Data** (ML).
  - Using physics to constrain learning, and learning to accelerate physics.


---

# II. Numerical Foundations (Ch. 21-22)

### 1. Numerical Integration
- **Trapezoid & Simpson's Rule**: Discrete sums of continuous areas.
- **Forward Link**: Essential for computing the **Kuramoto Order Parameter ($r$)**.

### 2. ODE Initial Value Problems
- **Euler's method**: Simple, but unstable for complex dynamics.
- **Runge-Kutta (RK4)**: The "Gold Standard" — 4 weighted slope samples.
- **The JAX Factor**: Implementing RK4 in JAX makes the *entire* simulation **differentiable**.

---

# III. The ODE Challenge: Kuramoto Model

### 1. Intuition: The Symphony of Spontaneity
- Fireflies synchronizing; audience clapping in unison.
- **Emergence**: Simple local rules $\to$ Sudden global order.

![bg right:20% height:90%](./Screenshot(3).png)

---

### 2. The Mathematical Engine
$$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} \sin(\theta_j - \theta_i)$$

- **The Bottleneck**: $O(N^2)$ pairwise interactions.

```python
def f(th): 
    # Broadcasting: (1, N) - (N, 1) = (N, N) matrix
    return omega + (K/N) * np.sin(th[None, :] - th[:, None]).sum(axis=1)
```

- **Order Parameter ($r$)**: $r \approx 0$ (Chaos) $\to$ $r \approx 1$ (Sync).

$$Z = re^{i\psi} = \frac{1}{N} \sum_{j=1}^N e^{i\theta_j}$$

![bg right:20% height:90%](./Screenshot(3).png)

---
```python
# Core of NumPy version
for i in range(steps):
    # Calculate Order Parameter r (Current state)
    z = np.mean(np.exp(1j * theta))
    r_history_np[i] = np.abs(z)
    
    # Standard RK4 math
    k1 = f(theta)
    k2 = f(theta + dt/2 * k1)
    k3 = f(theta + dt/2 * k2)
    k4 = f(theta + dt * k3)
    
    # Update theta for the next iteration
    theta = theta + (dt/6) * (k1 + 2*k2 + 2*k3 + k4) 
```
---
```python
# Core of JAX version: replace for loop by lax.scan
def simulation_step(theta, _):
    #---This block is similar to NumPy version---
    k1 = f(theta)
    # ...etc
    next_theta = theta + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    #---This block is similar to NumPy version---
    
    # Return (state_for_next_step, value_to_save_in_history)
    return next_theta, r
@jit
def run_full_simulation(initial_theta):
    # lax.scan(function, initial_state, sequence_of_steps)
    final_theta, r_history =lax.scan(simulation_step, initial_theta, jnp.arange(steps))
    return final_theta, r_history
```
---
### 3. Difference Between JAX and Numpy versions.

In the JAX version:

- **JIT into One Kernel**

- **lax.scan Replaces for loop:** 
1.  Fused --> Faster.
2. Prevents unrolling the 2000-step for loop.

---

# III. Why NumPy Fails?

- **Atomic Kernels**: every operation needs to visit the RAM back-and-forth.
- **Memory Bandwidth**: Large intermediate $N \times N$ matrices in NumPy.

### The JAX Payoff:
- **XLA**: Compiles Python math into one optimized XLA mega-kernel.
- **`lax.scan`**: Prevents JAX from unrolling and compiling massive python for loop.
- **Scale**: $N=10,000$ oscillators handled as easily as $N=100$.

---
**Time cost on laptop (i5-8250U CPU):**
| 		|JAX+jit+scan|Numpy|
|:------	|:------	|:------	|
|N = 100	|1.0629s	|1.0103s	|
|N = 1000	|51.1063s	|134.5051s	|
|N = 10000	|5362.2742s	|13673.8155s	|

**Time cost on Colab (JAX uses T4 GPU):**
| 		|JAX+jit+scan with compile|Kuramoto_Numpy|
|:------	|:------	|:------	|
|N = 100	|0.4526s	|1.6549s	|
|N = 500	|1.0035s	|62.6158s	|
|N = 10000	|6.7081s	|	|

<!-- Numpy uses sever CPU when running on Colab, so it performed even worse than using laptop CPU. -->

---

# IV. Extending to PDEs: Method of Lines

### 1. Connecting the Story

**The PDE part continues the same story as the ODE part:**
1. Move from **ODEs** to **PDEs**
2. Convert the PDE into an ODE system by **Method of Lines**
3. Reuse the same **JAX RK4 + `lax.scan`** pipeline
4. Differentiate through the solver with `jax.grad`
5. Extend from classical solvers to **PINNs**

**Main message:** PDEs can be brought into the same SciML pipeline as ODEs.

---

### 2. Problem Setup: 1D Heat Equation

We use one simple PDE throughout the whole section:
$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}, \qquad x \in [1],\ t \ge 0.
$$

**Boundary and initial conditions:**
$$
u(0,t)=0, \qquad u(1,t)=0, \qquad u(x,0)=\sin(\pi x).
$$

**For this choice, the exact solution is known:**
$$
u(x,t)=e^{-\alpha \pi^2 t}\sin(\pi x).
$$
---
**Why this example?**
- Physically intuitive: diffusion smooths and damps the profile.
- Easy to discretize and solve numerically.
- Has an exact solution for validation.
- Can be reused for benchmarks, differentiable programming, and PINNs.

---

### 3. Exact solution check: the profile should decay over time

![Exact Solution Check](https://hackmd.io/_uploads/rJRSU6BpZe.png)

---

**Interpretation:**
- The initial peak is about 1.
- At final time $T=0.2$, the profile is lower, around $e^{-\alpha\pi^2 T}$ times the original amplitude.
- This is exactly what diffusion should do: **same shape family, smaller amplitude**.

**Conclusion:** This result is physically and mathematically correct.

---

### 4. Method of Lines: Turning the PDE into ODEs

**Step 1: Discretize space**
Let interior grid values be
$$
U(t) = [u(x_1,t),u(x_2,t),\dots,u(x_M,t)]^\top.
$$
Approximate the second derivative by the central difference:
$$
\frac{\partial^2 u}{\partial x^2}(x_i,t) \approx \frac{u_{i+1}(t)-2u_i(t)+u_{i-1}(t)}{h^2}.
$$
---
**Step 2: Obtain a semidiscrete ODE system**
$$
\frac{dU}{dt} = \frac{\alpha}{h^2} A U,
$$
where $A$ is the discrete Laplacian matrix.

**The PDE is not solved all at once; after spatial discretization, it becomes a coupled ODE system, so we can reuse the RK4 machinery from the ODE part.**

---

### 5. Same Solver Pipeline as the ODE Group

**Computational pipeline:**
PDE $\rightarrow$ spatial discretization $\rightarrow$ ODE system $\rightarrow$ RK4 $\rightarrow$ JAX acceleration

**Important connection to the ODE part:**
- In Kuramoto, the ODEs come from oscillator interactions.
- In the heat equation, the ODEs come from spatial discretization.
- Computationally, both become: a state vector, a right-hand side, and a time integrator.

---

**Why I intentionally used the same style as my groupmates:**
- JAX implementation uses **RK4 + `lax.scan` + `jit`**.
- This keeps the presentation coherent.
- It reinforces the main SciML idea: **one computational template can cover both ODEs and PDEs.**

---

# V. Numerical Validation & Benchmarks

### 1. NumPy MOL Solver vs Exact Solution

![NumPy Baseline](https://hackmd.io/_uploads/HyUu8THabe.png)

---

**Observed errors at final time:**
- Max error $\approx 3.25 \times 10^{-5}$
- $L^2$ error $\approx 2.32 \times 10^{-5}$

**Interpretation:**
- The numerical solution is visually indistinguishable from the exact one.
- The error magnitude is very small for a classroom-scale experiment.
- This confirms that the Method of Lines + RK4 baseline is implemented correctly.

**Conclusion:** The NumPy baseline is reliable enough to serve as the classical reference solution.

---

### 2. JAX Version vs NumPy vs Exact Solution

![JAX vs NumPy](https://hackmd.io/_uploads/Sy_cL6r6-e.png)

---

**Observed JAX errors at final time:**
- Max error $\approx 3.25 \times 10^{-5}$
- $L^2$ error $\approx 2.31 \times 10^{-5}$

**Interpretation:**
- JAX and NumPy agree extremely well.
- Both also agree with the exact solution.
- This means the JAX implementation preserves correctness while changing the execution model.

**JAX is not changing the mathematics; it is changing how the same solver is executed and optimized.**

---

### 3. Physical Interpretation Over Time

![Diffusion Profiles](https://hackmd.io/_uploads/BJP3LpHpbx.png)

---
**What this figure shows:**
- The solution stays smooth.
- The peak decreases over time.
- Boundary values remain near zero.
- The heat profile is damped by diffusion.

---

### 4. Single-Solve Benchmark

![Single Benchmark](https://hackmd.io/_uploads/SJ86IprTZl.png)

---
**Recorded timings:**
- $M=63$: NumPy 0.001004 s, JAX 0.000159 s, speedup $\approx 6.31\times$
- $M=127$: NumPy 0.006363 s, JAX 0.000973 s, speedup $\approx 6.54\times$
- $M=255$: NumPy 0.035710 s, JAX 0.015477 s, speedup $\approx 2.31\times$

**Interpretation:**
- JAX is already competitive even on CPU.
- For this implementation, `jit + lax.scan` reduces Python overhead significantly.
- Speedup is not monotone and should not be overclaimed: it depends on problem size, hardware, and timing methodology.

**For repeated solver execution, JAX gives a meaningful constant-factor improvement while preserving accuracy.**

---

### 5. Batched-Solve Benchmark

![Batch Benchmark](https://hackmd.io/_uploads/BklALaBabg.png)

---
**Recorded timings:**
- Batch 8: speedup $\approx 5.61\times$
- Batch 32: speedup $\approx 12.23\times$
- Batch 128: speedup $\approx 25.73\times$

**Interpretation:**
- This is the clearest JAX advantage in my PDE section.
- NumPy loops over many solves explicitly.
- JAX uses `vmap + jit` to push the batch into one compiled array program.
- The larger the batch, the more clearly the JAX advantage appears.

**Integration with the ODE group:** This mirrors the Kuramoto story: JAX is especially strong when we move from one solve to many solves.

---

# VI. Differentiable Programming

### 1. Recover the Diffusion Coefficient

![Alpha Recovery](https://hackmd.io/_uploads/rJPkw6SpWe.png)

---
**Setup:**
- True coefficient: $\alpha_{\text{true}} = 0.15$
- Initial guess: $\alpha_0 = 0.05$
- Loss compares the solver output at final time against a target profile.
- `jax.grad` differentiates **through the whole RK4 time-stepping loop**.

**Interpretation:**
- The estimated $\alpha$ converges from 0.05 to about 0.15.
- The loss decreases to essentially zero.
- This is a strong demonstration that the classical solver is differentiable end-to-end.

**Once the solver is differentiable, simulation becomes optimizable.**

---

### 2. Why This Matters for SciML

**Classical numerical analysis gives us:**
- discretization,
- stability-aware time stepping,
- interpretable structure.

**JAX adds:**
- automatic differentiation,
- vectorization (`vmap`),
- compiled execution (`jit`).

**Combined result:**
We can now do more than just solve forward problems: parameter estimation, inverse problems, gradient-based calibration, and eventually neural surrogates such as PINNs.

---

# VII. From Classical Solver to PINNs

### 1. The Shift in Viewpoint

**Classical solver viewpoint:**
- Unknowns are values on a mesh.
- Derivatives are approximated by finite differences.
- Error analysis is classical and explicit.

**PINN viewpoint:**
Represent the solution by a neural network: $u_\theta(x,t)$.
Train it by minimizing:
$$
\mathcal{L} = \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{BC}} + \mathcal{L}_{\text{IC}}.
$$

---
**Interpretation:**
- Instead of solving for grid values, we learn a function.
- Automatic differentiation gives $u_t$ and $u_{xx}$ directly.
- This is why the JAX autodiff story naturally leads into PINNs.

---

### 2. PINN Training Behavior

![PINN Loss](https://hackmd.io/_uploads/By3lwTB6-e.png)

---
**Interpretation:**
- The overall trend is strongly downward.
- There are several spikes during training.
- This is not unusual for PINNs, especially with higher-order derivatives and Adam-style optimization.

**Spikes:** The PINN optimization is somewhat noisy, but the final loss is low and the final prediction is accurate.

---

### 3. PINN Final Result vs Exact Solution

![PINN vs Exact](https://hackmd.io/_uploads/H1QGP6rTWe.png)

---
**Measured final-time MSE:**
- PINN MSE vs exact final profile $\approx 1.51 \times 10^{-4}$

**Interpretation:**
- The learned solution is very close to the exact solution.
- Visually the agreement is strong over the whole domain.
- The final MSE is small enough to support a classroom demonstration.

**Honest comparison to the classical solver:**
- The classical MOL solver is still more direct and reliable for this simple forward PDE.
- The PINN is attractive because it is mesh-free and fits naturally into inverse or data-constrained settings.

---

# VIII. Conclusions on The PDE part

### 1. Classical Solver vs PINN

| Aspect | Method of Lines + RK4 | PINN |
|---|---|---|
| **Representation** | Grid values | Neural network function |
| **Derivatives** | Finite differences | Automatic differentiation |
| **Forward solve** | Strong and reliable | Usually slower to train |
| **Inverse problems** | Need gradients/adjoints | Natural to incorporate |
| **Best use here** | Reference numerical solver | SciML extension / learning viewpoint |

**These two approaches are not competitors in this talk; they are two stages of the same SciML story.**

---

### 2. Overall Evaluation

**Overall evaluation of results:**
- The heat equation example is solved correctly.
- NumPy and JAX agree with the exact solution.
- Single-solve and batched benchmarks both show meaningful JAX benefit.
- Differentiable parameter recovery works very well.
- The PINN reaches a low final error and matches the exact final profile closely.

**Three main conclusions:**
1. **PDEs can enter the same JAX pipeline as ODEs** through Method of Lines.
2. **JAX provides both speed and differentiability** for classical numerical solvers.
3. **PINNs are a natural SciML extension** once differentiation through physics is available.
---
**My part shows how the presentation moves from classical numerical PDE solving to differentiable scientific machine learning, while staying fully consistent with the ODE group's JAX-based solver framework.**

---
# IX. Summary

### 1. From Bottlenecks to Benchmarks
* **Overcoming Complexity**: JAX transformations (`jit` + `lax.scan`) bypassed the $O(N^2)$ NumPy bottleneck, turning a 3.8-hour laptop simulation into a 6.7-second GPU execution.
* **The Challenge:** Classical $O(N^2)$ Kuramoto interactions and PDE discretizations are computationally expensive in Python.
* **The JAX Solution:** By using `jit` for kernel fusion, `lax.scan` optimized loops, and GPU, we achieved a ~2,000x speedup on $N=10,000$ systems compared to laptop NumPy.
---
### 2. The Unified Computational Template

* **The Bridge:** The Method of Lines allows us to treat PDEs as systems of ODEs.

* **Consistency:** This enables the reuse of the same high-performance **RK4 + `lax.scan`** pipeline across diverse physical domains.

---
### 3. Solving $\to$ Differentiating $\to$ Learning
* **Differentiable Solvers**: jax.grad allows us to differentiate through entire simulation loops to recover hidden physical parameters like $\alpha$.
* **The PINN Shift**: Advanced from grid-based solvers to mesh-free neural representations, effectively fusing classical domain knowledge with modern machine learning.

* **Final Takeaway**: JAX turns classical numerical methods into a high-speed, differentiable language, making the transition from traditional solvers to Scientific Machine Learning seamless and scalable.