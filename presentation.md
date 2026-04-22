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

![bg right:20% height:90%](Screenshot(3).png)

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

![bg right:20% height:90%](Screenshot(3).png)

---
```python
# NumPy version
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
# JAX version: replace for loop by lax.scan
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
- **`jit`**: Compiles Python math into one optimized XLA mega-kernel.
- **`lax.scan`**: Prevents JAX from unrolling and compiling massive python for loop.
- **Scale**: $N=10,000$ oscillators handled as easily as $N=100$.

---
### Final Result:
| 		|JAX+jit+scan|Numpy|
|:------	|:------	|:------	|
|N = 100	|1.0629s	|1.0103s	|
|N = 500	|9.4828s	|26.3962s	|
|N = 1000	|51.1063s	|134.5051s	|
|N = 5000	|1428.3713s	|4101.0430s	|
---

# IV. Extending to PDEs: Method of Lines

### 1. Spatial Discretization
- 1D Heat Equation: $\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$
- Discretize space into $M$ grid points.
- Use **Central Finite Difference** for $\frac{\partial^2 u}{\partial x^2}$.

### 2. The ODE Bridge
- Discretized space + Continuous time = **A system of coupled ODEs**.
- **Result**: We can use the *same* RK4 solver from Section II!

---

# V. Unified JAX Showcase: Benchmarks

| Method | Scaling Complexity | JAX Optimization |
| :--- | :--- | :--- |
| **Kuramoto** | $O(N^2)$ Pairwise | `jit`+`lax.scan` |
| **Heat Eq.** | $O(M)$ Grid points | Differentiable RK4 |

### The Power of `jax.grad`
- We can differentiate the *entire* time-stepping loop.
- Compute $\frac{\partial (\text{Loss})}{\partial \alpha}$ to find the diffusion coefficient.
- **This is Differentiable Programming.**

---

# VI. From Solvers to Learners: PINNs

**Physics-Informed Neural Networks** represent $u(x, t)$ as a network $u_\theta(x, t)$.

### The Loss Function:
$$L = L_{PDE} + L_{BC} + L_{IC}$$

1. **$L_{PDE}$**: Residual of the heat equation at random points.
2. **$L_{BC} / L_{IC}$**: Constraints at boundaries and $t=0$.

**No Mesh Required**: Uses "Collocation Points" instead of a rigid grid.

---

# Classical Solver vs. PINN

| Aspect | Method of Lines | PINN |
| :--- | :--- | :--- |
| **Representation** | Grid-based | Neural Network $u_\theta(x, t)$ |
| **Mesh** | Required | **Mesh-free** |
| **Inverse Problems** | Complex Adjoint Methods | **Natural/Easy** |
| **Error Control** | Well-understood | Harder to guarantee |

![bg right:35% width:95%](https://bazhenov.me/images/pinn.png)

---

# VII. Conclusion & SciML Synthesis

- **Integrate $\to$ Solve $\to$ Scale $\to$ Differentiate $\to$ Learn.**
- **Foundations Matter**: Numerical stability (RK4) is the engine of AI physics.
- **JAX**: Transforms Python code into GPU-ready mathematical hardware.
- **The Future**:
  - Neural ODEs (Solving and learning ODEs simultaneously).
  - Operator Learning (DeepONet / FNO).

---

# VIII. Q&A

### Thank you for your attention!

- *Why RK4 over adaptive methods?*
- *How does `lax.scan` differ from for loops?*
- *Limitations of PINNs in chaotic systems?*

**Open for discussion.**