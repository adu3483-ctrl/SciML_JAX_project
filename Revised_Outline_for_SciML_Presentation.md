---
title: Outline for SciML Presentation
---

## Topic: Introduction to Scientific Machine Learning — From Numerical Foundations to Physics-Informed Learning
**Course:** Numerical Computation with JAX
**Members:**
- 梁謹 Liang Chin & 廖品維 Liao Pin Wei (ODE / Kuramoto Group)
- 陳元龍 Chen Yuan Lung (PDE / PINN Group)

**Target Duration:** 45–50 Minutes (including Q&A)

---

## Narrative Arc

> **One story in five acts:**
> Solve ODEs → Scale them with JAX → Extend to PDEs via Method of Lines →
> Accelerate PDE solvers with JAX → Replace classical solvers with learned models (PINNs).
>
> **Driving question:**
> *How do we go from a textbook differential equation to a high-performance,
> differentiable simulation — and what happens when we let the machine learn the physics itself?*

---

## I. Introduction — Why Scientific Machine Learning? (3 min)

> **Presenter:** ODE Group (Liang Chin or Liao Pin Wei)

* **The Bridge Between Theory and Simulation**
    * Science writes laws as differential equations; computers solve them by discretization.
    * Classical numerical methods (finite differences, RK4) are well-understood but hit walls at scale: high-dimensional systems, inverse problems, incomplete physics.
    * **SciML** fuses domain knowledge (PDEs, conservation laws, symmetries) with data-driven ML — using physics to constrain learning, and learning to accelerate physics.

* **Where This Talk Fits in the SciML Landscape**
    * The reference SciML curriculum (Krishna Kumar, UT Austin) covers 12 modules: MLPs, PINNs, Neural ODEs, DeepONet, FNO, GNNs, SINDy, Bayesian methods, etc.
    * This presentation focuses on the foundational pipeline: **numerical integration → ODE solvers → PDEs via Method of Lines → PINNs** — the first three modules of that curriculum, implemented in JAX.
    * Other SciML techniques (operator learning, equation discovery, Neural ODEs) are noted as extensions but not covered in depth.

* **The Roadmap** — preview the five-act structure:
    1. Numerical foundations (integration, ODE solvers)
    2. A real ODE challenge at scale (Kuramoto model)
    3. Extending to PDEs (Method of Lines)
    4. JAX as the common accelerator
    5. From solvers to learners (PINNs)

---

## II. Numerical Foundations (10 min)

> **Presenter:** ODE Group
>
> *Covers content from Chapters 21–22. Shared groundwork for both projects.*

### 1. Numerical Integration (Ch. 21) — 4 min

* **Core idea:** turning a continuous integral into a discrete sum of areas.
* **Key algorithms:**
    * Trapezoid Rule — linear approximation between sample points.
    * Simpson's Rule — quadratic fit, higher accuracy for smooth functions.
* **Forward link:** the Kuramoto order parameter ($r$) is computed as an integral over the system state; PDE spatial discretization also relies on these quadrature ideas.

### 2. ODE Initial Value Problems (Ch. 22) — 6 min

* **Setup:** $dS/dt = F(t, S)$ with initial condition $S(t_0) = S_0$.
* **Progression of accuracy:**
    * **Euler's Method** — the straight-line guess; simple but accumulates error and is often unstable.
    * **Predictor-Corrector** — peek at the future slope, then average; improved stability.
    * **Runge-Kutta (RK4)** — four weighted slope samples per step; the gold standard for non-stiff problems.
* **Key concept for later: differentiability.** If the time-stepper is implemented in JAX, the entire simulation becomes differentiable via `jax.grad` — this is the gateway to both parameter optimization and PINNs.
* **Forward link:** every oscillator in the Kuramoto model is an ODE; after Method of Lines, every grid point in the heat equation is also an ODE. RK4 drives both.

### Work Distribution
| Task | Owner |
|------|-------|
| Prepare slides and examples for Ch. 21–22 | ODE Group |
| Review for consistency with PDE terminology | PDE Group |

---

## III. The ODE Challenge at Scale: Kuramoto Model (8 min)

> **Presenter:** ODE Group

### 1. Intuition: The Symphony of Spontaneity — 2 min

* **Analogy:** fireflies synchronizing their flashing, an audience gradually clapping in unison.
* **Mechanism:** each oscillator has a natural frequency but is pulled by its neighbors.
* **Phase transition:** increasing coupling strength ($K$) causes a sudden snap from disorder to synchrony — an emergent phenomenon from simple local rules.

### 2. The Mathematical Engine — 2 min

* **Governing equation:**
$$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} \sin(\theta_j - \theta_i)$$
* **The $O(N^2)$ bottleneck:** every oscillator must "talk" to every other — the pairwise coupling sum dominates cost.
* **Order parameter** as the observable:
$$r\, e^{i\psi} = \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j}$$
* $r \approx 0$: incoherent (random phases). $r \approx 1$: fully synchronized.

### 3. Why Pure Python Fails — 1 min

* Nested `for` loops over $N^2$ pairs introduce massive interpreter overhead.
* NumPy helps, but large intermediate arrays ($N \times N$ phase-difference matrix) create memory-bandwidth bottlenecks for large $N$.

### 4. The JAX Solution (Preview) — 3 min

* **Broadcasting:** replaces loops with a single outer-subtraction matrix operation.
* **`jit` compilation (XLA):** fuses subtraction → sin → summation into one optimized kernel.
* **`vmap`:** runs many coupling-strength ($K$) sweeps in parallel — one line replaces an outer loop.
* **Teaser result:** JAX handles 10,000 oscillators in the time NumPy handles 100.

> *Detailed benchmarks are deferred to the unified comparison in Section V.*

### Work Distribution
| Task | Owner |
|------|-------|
| Kuramoto slides, code, and animations | ODE Group |
| Provide feedback on JAX narrative consistency | PDE Group |

---

## IV. Extending to PDEs: Method of Lines (7 min)

> **Presenter:** PDE Group (Chen Yuan Lung)

### 1. From ODEs to PDEs — 2 min

* ODEs describe evolution in time for finitely many variables.
* PDEs describe systems varying in both space *and* time (heat diffusion, wave propagation, fluid flow).
* **Main example:** 1D heat equation $\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$ — linear, smooth, ideal for introducing discretization.

### 2. Spatial Discretization — 2 min

* Replace the continuous spatial domain with $M$ grid points.
* Approximate the second spatial derivative by a central finite difference.
* After discretization, the PDE becomes a system of $M$ coupled ODEs — one per grid point.

### 3. The Method of Lines Bridge — 3 min

* **Key idea:** discretize space, keep time continuous → a semidiscrete ODE system.
* This system is solved with the *same* RK4-class integrators from Section II.
* **Semidiscrete vs. fully discrete:** separating spatial approximation from time integration gives modularity — you can swap spatial schemes or time integrators independently.
* **Connection to Kuramoto:** both projects now live inside the same "ODE system + time integrator" pipeline. The only difference is where the ODEs come from (phase dynamics vs. spatial discretization).
* **Connection to SciML:** once the solver is written in JAX, the entire simulation is differentiable — you can compute $\partial(\text{loss at final time})/\partial(\text{diffusion coefficient})$ via `jax.grad`. This is **differentiable programming**, Module 5 of the SciML curriculum.

### Work Distribution
| Task | Owner |
|------|-------|
| Prepare PDE / Method of Lines slides and examples | PDE Group |
| Ensure RK4 notation is consistent with Section II | ODE Group |

---

## V. Unified JAX Showcase: Benchmarks & Differentiation (10 min)

> **Presenters:** Both groups (split as indicated)
>
> *The merged "JAX payoff" section. Instead of each project showing benchmarks independently, we present a single coherent demonstration.*

### 1. Kuramoto Benchmarks — ODE Group — 4 min

* **Implementations compared:** Python loop → NumPy → JAX → JAX + `jit` → JAX + `jit` + `vmap`.
* **Speedup graphs:** wall-clock time vs. $N$ (number of oscillators).
* **Visualizing sync:** order parameter ($r$) climbing from 0 → 1 as coupling $K$ increases.
* **Key takeaway:** `jit` gives a constant-factor speedup; `vmap` enables qualitatively new experiments (sweeping $K$ in parallel).

### 2. Heat-Equation Benchmarks — PDE Group — 4 min

* **Same progression of implementations** applied to the semidiscrete heat-equation ODE.
* **Two tasks:** single solve vs. batched solves (many diffusion coefficients / initial conditions).
* **Key insight:** for a single small problem, JIT overhead dominates; for batched runs, `jit` + `vmap` win decisively.
* **Shared observation:** both Kuramoto and heat equation show the same JAX scaling pattern — validating the unified pipeline.

### 3. Differentiable Computation — PDE Group — 2 min

* Define a scalar loss at final time (e.g., deviation from a target temperature profile).
* Use `jax.grad` to differentiate the loss through the entire time-stepping loop w.r.t. the diffusion coefficient $\alpha$.
* **Punchline:** JAX provides not just speed, but automatic gradients through classical solvers — the gateway from **solving** to **learning**.
* **SciML connection:** this is exactly what PINNs exploit — if computing PDE residuals is differentiable, a neural network can be trained to minimize them.

### Work Distribution
| Task | Owner |
|------|-------|
| Kuramoto benchmark code, speedup plots, sync animations | ODE Group |
| Heat-equation benchmark code, single vs. batch plots | PDE Group |
| `jax.grad` demo on heat equation | PDE Group |
| Agree on shared plot style, axis labels, color scheme | Both (coordinate) |

---

## VI. From Solvers to Learners: PINNs (7 min)

> **Presenter:** PDE Group (Chen Yuan Lung)

### 1. The Core Idea — 2 min

* Instead of discretizing the domain onto a grid, represent the solution $u(x, t)$ as a neural network $u_\theta(x, t)$.
* The network is trained by minimizing a composite loss:
    * **PDE residual loss:** evaluate $\frac{\partial u_\theta}{\partial t} - \alpha \frac{\partial^2 u_\theta}{\partial x^2}$ at random collocation points; penalize non-zero residual.
    * **Boundary condition loss:** enforce known values at domain edges.
    * **Initial condition loss:** enforce known values at $t = 0$.
* Automatic differentiation (the same `jax.grad` from Section V) computes the PDE derivatives of the network output — no finite differences needed.

### 2. Key Training Considerations — 2 min

* **Collocation point sampling:** random points in the space-time domain; denser near boundaries or regions of rapid change. Adaptive strategies (e.g., residual-based refinement) improve convergence.
* **Loss balancing:** the PDE residual, BC, and IC terms can have very different magnitudes. Adaptive weighting schemes (e.g., learning-rate-based or gradient-statistics-based) prevent one term from dominating.
* **Soft vs. hard boundary constraints:** soft = penalty in the loss (simple but approximate); hard = build the constraint into the network architecture (exact but requires problem-specific design).

### 3. Classical Solver vs. PINN — 1 min

| Aspect | Classical (Method of Lines) | PINN |
|:---|:---|:---|
| Representation | Grid-based, pointwise values | Function-based, neural network |
| Mesh | Required (grid points) | Mesh-free (collocation points) |
| Error control | Well-understood (truncation error, stability) | Harder to guarantee (training convergence) |
| Inverse problems | Requires adjoint methods or finite-diff gradients | Natural — just add data loss terms |
| High dimensions | Curse of dimensionality | Scales more gracefully |
| JAX role | Accelerates the ODE system | Provides autodiff for PDE residuals + training |

### 4. Simple JAX-Based PINN Demo — 2 min

* Live or recorded demo: training a small MLP-based PINN on the 1D heat equation.
* Show loss curve converging; compare final $u_\theta(x, t)$ to the classical solver output from Section V.
* **Closing the loop:** the same equation solved two ways — one classical, one learned — both powered by JAX.

### Work Distribution
| Task | Owner |
|------|-------|
| PINN slides, code, and demo | PDE Group |
| Review and ask "audience-perspective" questions | ODE Group |

---

## VII. Conclusion & Synthesis (3 min)

> **Presenter:** One from each group (1.5 min each)

* **ODE Group:**
    * Foundations matter — you cannot build a fast simulation without a stable numerical engine (RK4, quadrature).
    * JAX transforms a Python bottleneck into GPU-scale computation via `jit` + `vmap`.
    * The Kuramoto model demonstrates that large coupled ODE systems are natural targets for JAX's array-oriented paradigm.

* **PDE Group:**
    * Method of Lines lets PDEs reuse the entire ODE pipeline — spatial discretization feeds into the same integrators.
    * Differentiable solvers (`jax.grad` through time-stepping) are the bridge between classical numerics and machine learning.
    * PINNs show what becomes possible when the solver itself is differentiable — the network learns physics from the equation, not from data.

* **Joint closing: The SciML arc**
    * **Integrate → Solve → Scale → Differentiate → Learn.**
    * This presentation covered the first three modules of the SciML curriculum. The natural next steps: Neural ODEs (Module 4), operator learning (DeepONet/FNO, Modules 6–7), equation discovery (SINDy, Module 11).
    * Synchronization and diffusion aren't just math problems — they are windows into how nature organizes itself, and we can now observe them in real time through code.

---

## VIII. Q&A (5 min)

> **Both groups on stage**

Possible discussion topics:
* Why RK4 over adaptive methods (e.g., Dormand-Prince) for this project?
* How does JAX's `vmap` differ from simple parallelism (e.g., multiprocessing)?
* Semidiscrete vs. fully discrete — when does the choice matter?
* What are collocation strategies for PINNs in higher dimensions?
* Could PINNs be applied to the Kuramoto model? (stretch question linking both projects)
* What are the limits of PINNs for stiff or chaotic systems?
* How do PINNs relate to other SciML approaches like DeepONet or Neural ODEs?

---

## Appendix: Collaboration Plan

### Shared Responsibilities
| Item | Details |
|------|---------|
| Slide template & style guide | Agree on fonts, colors, code-snippet style before Week 1 |
| Notation consistency | Align symbols ($S$, $\theta$, $u$, $F$, $\alpha$) across both projects |
| Shared JAX utility code | Common RK4 stepper, timing harness, plot style functions |
| Rehearsal | At least one full dry-run together to smooth transitions |

### Timeline Suggestion
| Week | ODE Group | PDE Group | Joint |
|------|-----------|-----------|-------|
| 1 | NumPy/SciPy Kuramoto baseline | Heat-equation spatial discretization | Agree on notation, slide template |
| 2 | JAX Kuramoto + benchmarks | JAX heat-equation + `grad` demo | Share JAX utility code; mid-point review |
| 3 | Sync visualizations, polish slides | PINN implementation + demo | Merge slides; rehearse full presentation |

---

## Changes from Original Outline

1. **Added SciML curriculum context** — Section I now situates this presentation within the broader SciML landscape (referencing the 12-module curriculum), explaining which modules are covered and which are left as extensions. This helps the audience understand *where* this talk fits in the field.
2. **Added differentiability as a recurring thread** — Section II now explicitly notes that implementing RK4 in JAX makes the simulation differentiable, and this idea is threaded through Sections IV, V, and VI as the connecting tissue between classical numerics and PINNs. The original outline introduced `jax.grad` only in Section V without foreshadowing.
3. **Expanded the Method of Lines section** — added the explicit connection to differentiable programming (Module 5 of SciML curriculum) and clarified the semidiscrete vs. fully discrete distinction with a modularity argument.
4. **Enriched the PINN section with training considerations** — added collocation point sampling strategies, adaptive loss weighting, and soft vs. hard boundary constraints. These are core topics in the SciML curriculum (Module 3) that the original outline only hinted at.
5. **Added a Classical vs. PINN comparison table** — concrete side-by-side on representation, mesh requirements, error control, inverse problems, dimensionality, and JAX's role. Replaces the original's prose comparison.
6. **Strengthened the JAX benchmark section** — added a "key takeaway" for each benchmark subsection clarifying *what* the speedup means (constant-factor vs. qualitatively new experiments), and noted the shared scaling pattern between Kuramoto and heat equation.
7. **Added SciML "next steps" to the conclusion** — explicitly points to Neural ODEs, DeepONet/FNO, and SINDy as the natural continuation, connecting this presentation to the rest of the curriculum.
8. **Expanded Q&A suggestions** — added questions about collocation strategies, PINN limitations for stiff/chaotic systems, and how PINNs relate to other SciML approaches (DeepONet, Neural ODEs).
9. **Preserved the original's strong structure** — the five-act narrative arc, the unified benchmark section, the collaboration plan, and the work distribution tables were all well-designed and retained with refinements.
