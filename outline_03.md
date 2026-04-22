# Presentation Outline: Scaling Synchronization
## From Numerical Foundations to High-Performance Modeling with JAX

**Members:** 梁謹 Liang Chin, 廖品維 Liao Pin Wei

**Target Duration:** 30 Minutes

---

## I. Introduction (3 Minutes)
* **The Bridge Between Theory and Simulation**
    * Moving from analytical "pen-and-paper" solutions to numerical approximations.
    * Why modern science relies on discrete "steps" to model continuous reality.
* **The Roadmap**
    * Part 1: The "Guts" — Foundations of Integration and ODEs (Chapters 21–22).
    * Part 2: The "System" — An intuitive look at the Kuramoto Model.
    * Part 3: The "Speed" — Leveraging JAX to scale from 10 to 10,000 oscillators.

---

## II. Part 1: Numerical Foundations (12 Minutes)

### 1. Numerical Integration (Chapter 21)
* **The Core Concept:** Turning a continuous integral into a discrete sum of areas.
* **Key Algorithms:**
    * **Trapezoid Rule:** Linear approximation.
    * **Simpson’s Rule:** Quadratic approximation for higher accuracy.
* **Coherence Link:** In Kuramoto modeling, we don't just solve for one point; we calculate the **Order Parameter ($r$)**, which is essentially a spatial average (an integral) of the state of the entire system at a specific time.

### 2. ODE Initial Value Problems (Chapter 22)
* **The Setup:** $dS/dt = F(t, S)$ where we know the starting point.
* **The Evolution of Accuracy:**
    * **Euler’s Method:** The "straight-line" guess. Simple but unstable over long periods.
    * **Predictor-Corrector:** Taking a "peek" at the future slope to adjust the current step.
    * **Runge-Kutta (RK4):** The "Gold Standard." Using four weighted slope samples to stay perfectly on the curve.
* **Coherence Link:** Every single "firefly" in our Kuramoto model is an individual ODE. To simulate the group, we must run $N$ copies of these solvers simultaneously.

---

## III. Part 2: The Kuramoto Model (5 Minutes)

### 0. Why The Kuramoto model?

* **The $O(N^2)$ Bottleneck:** Loops vs. VectorizationThe Governing Equation:$$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^N \sin(\theta_j - \theta_i)$$
* **The Python Problem:** All-to-all interaction requires $N^2$ operations.Nested for loops in Python introduce massive interpreter overhead and GIL limitations.
* **The JAX Solution:** Broadcasting: Replaces loops with a single matrix operation (outer subtraction).

    * **Hardware Acceleration**: Dispatches vectorized tasks directly to GPUs/TPUs.
* **Intense Math & XLA Compiler OptimizationTranscendental Functions:** Calculating sin() millions of times is computationally expensive.
* **Memory Bandwidth Issue:** Standard NumPy creates large intermediate arrays, causing a "read/write" bottleneck in RAM. 
* **XLA (Accelerated Linear Algebra):**
    * **Operator Fusion:** Fuses subtraction, sine, and summation into a single machine-code kernel.Reduces memory overhead and optimizes non-linear function execution.
* **Scalability & Physical Significance** 
    * **Scaling:** As $N$ grows, Python follows a steep quadratic curve; JAX remains relatively flat. 
    * **Scientific Utility:** Calculating the Order Parameter ($r$) to observe phase transitions.$$r e^{i\psi} = \frac{1}{N} \sum_{j=1}^N e^{i\theta_j}$$
    * **The Verdict:** JAX transforms nature’s complex chaos into an optimized, high-performance simulation.

### 1. Intuition: The Symphony of Spontaneity
* **The Analogy:** Fireflies in a field or an audience clapping.
* **The Mechanism:** Oscillators have a "natural" speed but are influenced by their neighbors.
* **The Phase Transition:** Increasing "coupling strength" ($K$) causes a sudden snap from chaos to perfect unison.

### 2. The Mathematical Engine
* **The Equation:** $\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} \sin(\theta_j - \theta_i)$
* **The Complexity:** This is an $O(N^2)$ problem. To move one step forward, every oscillator must "talk" to every other oscillator.

---

## IV. Part 3: The 3-Week Project Showdown (8 Minutes)

### Week 1: The Baseline (NumPy & SciPy)
* Implementing the model using standard `scipy.integrate.solve_ivp`.
* Identifying the **Performance Wall**: Why NumPy starts to "choke" when we try to simulate 1,000+ interacting oscillators on a standard CPU.

### Week 2: The JAX Transformation
* **Vectorization:** Replacing slow Python `for` loops with high-speed matrix operations.
* **The JAX Superpowers:**
    * `jit` (Just-In-Time compilation): Turning Python code into optimized machine code (XLA).
    * `vmap`: Running 50 different simulations (different $K$ values) in parallel with one command.

### Week 3: Benchmarking & Visualization
* **The Data:** Creating "Speedup Graphs" (NumPy vs. JAX).
* **The Result:** Demonstrating that JAX can handle 10,000 oscillators in the time NumPy handles 100.
* **Visualizing Sync:** Showing the "Order Parameter" ($r$) climbing from 0 to 1 as the simulation runs.

---

## V. Conclusion & Summary (2 Minutes)
* **Foundations Matter:** You cannot build a fast simulation (JAX) without a stable numerical engine (RK4).
* **Hardware Empowerment:** Modern libraries allow us to take the math from the Berkeley textbook and run it at a "God-mode" scale on GPUs.
* **Final Thought:** Synchronization isn't just a math trick; it’s a fundamental property of nature that we can now observe in real-time through code.

---
---

## VI. Q&A
* *Open floor for technical questions on RK4 vs. JAX performance.*

---

# Presentation Outline: From PDE Numerical Methods to Scientific Machine Learning

## From Classical PDE Solvers to PDE Learning with JAX

**Member:** 陳元龍 Chen Yuan Lung

**Target Duration:** 30 Minutes  

---

## I. Introduction (2 Minutes)

- **Motivation**
  - ODEs describe evolution in time for finitely many variables.
  - PDEs describe systems varying in both space and time.
  - Many physical processes, such as heat diffusion and wave propagation, are modeled by PDEs.

- **Main Question**
  - How can a continuous PDE be transformed into a computable numerical system?
  - How does this connect with the group’s ODE/JAX/Neural ODE framework?

- **Outline**
  - Part 1: Classical PDE numerical foundation
  - Part 2: Method of Lines and JAX implementation
  - Part 3: PINN as a SciML comparison

---

## II. Part 1: Classical PDE Numerical Foundations (5 Minutes)

### 1. Main Example: 1D Heat Equation
- **Interpretation:** heat diffuses from hotter regions to colder regions.
- **Why this example:** linear, smooth, and suitable for introducing PDE discretization.

### 2. Spatial Discretization
- Replace the spatial interval by grid points.
- Approximate spatial derivatives by finite differences.

### 3. Numerical Meaning
- After discretization, the PDE becomes a coupled system for approximate values at grid points.
- This provides the bridge from a continuous PDE to a finite-dimensional numerical problem.

---

## III. Part 2: Method of Lines and the PDE-to-ODE Bridge (5 Minutes)

### 1. Method of Lines
- Discretize space and keep time continuous.
- The time-dependent PDE is transformed into a semidiscrete ODE system.

### 2. Why This Form Matters
- This is a standard numerical framework for time-dependent PDEs.
- It links PDE numerical analysis directly with ODE solvers.

### 3. Semidiscrete vs. Fully Discrete
- **Semidiscrete:** first obtain an ODE system, then apply a time integrator.
- **Fully discrete:** discretize both space and time immediately.
- The semidiscrete viewpoint separates spatial approximation from time integration.

### 4. Position in This Project
- In this project, JAX is used **after spatial discretization**.
- JAX is applied to the resulting ODE system, rather than to the original continuous PDE directly.

---

## IV. Part 3: JAX Implementation and Computational Comparison (8 Minutes)

### 1. Computational Setup
- Use the same semidiscrete heat-equation example throughout this part.
- Apply the same time integrator in all implementations.

### 2. Code Example
- Implement the same semidiscrete ODE system in two forms:
  - NumPy baseline
  - JAX version

- The comparison is made at the level of the **ODE system derived from the PDE**.

### 3. Comparison Setting
- Compare the following implementations:
  - NumPy loop
  - JAX implementation
  - JAX with `jit`
  - JAX with `jit` and `vmap`
- Compare them under two tasks:
  - **single solve**
  - **batched solves** for many diffusion coefficients or many initial conditions

### 4. Main Computational Goal
- JAX is used here to demonstrate:
  - acceleration through compilation,
  - batched computation,
  - differentiable computation on the semidiscrete ODE system.

### 5. What the Comparison Shows
- For a single small problem, JAX may not show a strong speed advantage because of compilation overhead.
- For repeated or batched simulations, `jit` and `vmap` make the JAX implementation more efficient and scalable.

### 6. Additional Computational Point
- Define a scalar loss at final time.
- Use `grad` to differentiate this loss with respect to a parameter such as the diffusion coefficient.
- This shows that JAX provides not only acceleration, but also differentiable computation for PDE-based models.

### 7. Link to the ODE/JAX Group
- Method of Lines converts the PDE into an ODE system.
- This places the PDE problem naturally inside the same computational pipeline as the ODE/JAX subgroup.

---

## V. Part 4: PINN as a SciML Comparison (8 Minutes)

### 1. PINN Formulation
- A Physics-Informed Neural Network represents the solution by a neural network.
- The training objective enforces:
  - small PDE residual,
  - boundary conditions,
  - initial condition.

### 2. Comparison with the Classical Solver
- **Classical numerical solver**
  - grid-based approximation,
  - strong baseline for forward problems.

- **PINN**
  - function-based approximation,
  - provides a SciML-based alternative representation of the solution.

### 3. JAX in the PINN Route
- In the PINN route, JAX is used for:
  - automatic differentiation of PDE-related derivatives,
  - construction of the residual-based loss,
  - optimization in a simple PINN implementation.

### 4. Position in This Project
- A simple JAX-based PINN case study is included as a comparison.

### 5. Computational Connection
- In the classical route, JAX is used for the semidiscrete ODE system obtained after spatial discretization.
- In the PINN route, JAX is used for autodiff and training.
- This gives two different uses of JAX within the same PDE example.

---

## VI. Optional Extension: Burgers Equation

- Burgers equation introduces nonlinearity.
- It extends the same pipeline from a linear PDE to a nonlinear PDE.
- It is included only as an optional or future extension.

---

## VII. Conclusion (2 Minutes)

- A time-dependent PDE can be transformed into a computable system through spatial discretization.
- Method of Lines provides the bridge from PDEs to ODE systems.
- In this project, JAX is applied to the semidiscrete ODE system derived from the PDE.
- A simple JAX-based PINN case study is also included as a SciML comparison.
- PINNs provide a learning-based alternative within the broader SciML framework.

---

## VIII. Q&A

- Possible discussion:
  - Why use the heat equation as the main example?
  - What is the advantage of the semidiscrete viewpoint?
  - Why is batched comparison a better setting for showing JAX’s advantage?
  - How do classical solvers and PINNs differ in representing the solution?
  - What are the two different roles of JAX in the classical route and in the PINN route?