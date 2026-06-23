# Computational Physics in Python

A curated, tutorial-style collection of numerical simulations, solvers, and visualisations for
classical, quantum, statistical, and condensed-matter physics — implemented from scratch in Python
with NumPy, SciPy, Matplotlib, PyTorch, and DeepXDE.

Every notebook follows the same four-part structure:

1. **Physical system** — what is being modelled, in plain English.
2. **Governing equations** — written in LaTeX with proper notation.
3. **Numerical method** — which discretisation, scheme, or learning algorithm is used and why.
4. **Results & plots** — annotated visualisations of the output.

---

## Repository structure

| Folder | Domain | Notebooks |
|---|---|---|
| [`math_methods/`](math_methods) | Numerical foundations | derivatives, integration, ODE solvers, Fourier methods, fitting, interpolation, Monte-Carlo, shooting method |
| [`classical_mechanics/`](classical_mechanics) | Newtonian dynamics | coupled-oscillator eigenmodes, 2-D harmonic motion, 3-body gravitation, Lorenz attractor |
| [`electrodynamics/`](electrodynamics) | Maxwell theory | magnetic field of a current-carrying wire |
| [`quantum_mechanics/`](quantum_mechanics) | Schrödinger equation | 1-D Schrödinger via the shooting method |
| [`statistical_mechanics/`](statistical_mechanics) | Stochastic & thermal | Heisenberg/Ising-style Metropolis MC, classical phase-space trajectories |
| [`pdes/`](pdes) | Partial differential equations | heat equation (1-D & 2-D), finite-element method (FEM) scripts |
| [`condensed_matter/`](condensed_matter) | Solid-state physics | graphene tight-binding band structure |
| [`machine_learning_physics/`](machine_learning_physics) | ML for physics | PyTorch primer & physics-informed neural networks (PINNs) for Burgers', heat, and Navier–Stokes equations |
| [`data/`](data) | Tabulated input data used by notebooks |  |

---

## Notebook index

### `math_methods/`
| Notebook | Topic |
|---|---|
| `derivatives_finite_difference.ipynb` | Forward/backward/central differences, error-vs-step study |
| `derivatives_theory_and_examples.ipynb` | Stencil derivation, Richardson extrapolation, gradient/divergence/curl |
| `derivatives_from_data.ipynb` | Numerical differentiation of an experimental time series |
| `integration_methods.ipynb` | Rectangle, trapezoidal, Simpson, Newton–Cotes |
| `integration_rotational_motion.ipynb` | Moments of inertia by Monte-Carlo volume sampling |
| `ode_solvers_euler_rk.ipynb` | Euler, RK4, RK45, SciPy `solve_ivp` comparison |
| `fitting_polynomial.ipynb` | Gradient-descent polynomial fitting with regularisation |
| `fitting_and_interpolation.ipynb` | Taylor, Lagrange, Chebyshev nodes, cubic splines |
| `fourier_transform_intro.ipynb` | Continuous FT by trapezoidal quadrature; intro to `np.fft` |
| `fft_implementation.ipynb` | DFT, FFT, STFT, Hartley, Haar wavelet — implemented from scratch |
| `monte_carlo_pi.ipynb` | Estimating π by dart-throw Monte-Carlo |
| `random_numbers_lcg_and_dice.ipynb` | LCG, seeding, dice, radioactive-decay chain |
| `random_numbers_nonuniform.ipynb` | Box–Muller, Rutherford scattering, MC integration |
| `shooting_method_bvp.ipynb` | Linear shooting for BVPs and 1-D quantum eigenstates |

### `classical_mechanics/`
| Notebook | Topic |
|---|---|
| `coupled_oscillators_eigenmodes.ipynb` | Normal modes via the eigenvalue problem |
| `rolling_ball_2d_harmonic.ipynb` | 2-D harmonic motion with optional damping and forcing |
| `three_body_gravitational.ipynb` | Sun–Earth–Moon + spaceship integration in SI units |
| `lorenz_attractor.ipynb` | Deterministic chaos and the butterfly effect |

### `electrodynamics/`
| Notebook | Topic |
|---|---|
| `magnetic_field_charged_wire.ipynb` | Biot–Savart vector potential, curl ⇒ B-field, analytical cross-check |

### `quantum_mechanics/`
| Notebook | Topic |
|---|---|
| `schrodinger_1d_shooting_method.ipynb` | Time-independent Schrödinger equation by shooting + bisection |

### `statistical_mechanics/`
| Notebook | Topic |
|---|---|
| `ising_model_monte_carlo.ipynb` | 3-D Heisenberg + Zeeman + DMI Metropolis sampler |
| `classical_phase_trajectories.ipynb` | Symplectic-Euler phase-space ensemble |

### `pdes/`
| Notebook / Script | Topic |
|---|---|
| `heat_equation_2d.ipynb` | Method-of-lines 1-D & 2-D heat equation |
| `finite_element_method/fem_1d_basic.py` | Bare-bones 1-D FEM |
| `finite_element_method/fem_1d_extended.py` | Extended 1-D FEM with boundary conditions |

### `condensed_matter/graphene_band_structure/`
| Notebook | Topic |
|---|---|
| `graphene_band_structure.ipynb` | Tight-binding electronic structure of graphene, Dirac cones |

### `machine_learning_physics/physics_informed_neural_networks/`
| Notebook | Topic |
|---|---|
| `00_pytorch_primer.ipynb` | PyTorch fundamentals refresher |
| `01_pinn_theory.ipynb` | What PINNs are and why they work |
| `02_burgers_equation_pinn.ipynb` | Viscous Burgers' equation with shock formation |
| `03_heat_equation_pinn.ipynb` | FDM vs. FEM vs. hand-coded PINN vs. DeepXDE |
| `04_navier_stokes_pinn.ipynb` | 2-D steady Navier–Stokes channel flow with DeepXDE |

---

## Setup

```bash
git clone https://github.com/dishantgupta2004/Computational_Physics_python.git
cd Computational_Physics_python
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
jupyter lab
```

The PINN notebooks additionally require `torch` and (for `03`/`04`) `deepxde`; install with
`pip install torch deepxde`.

---

## Adding a new simulation

To keep the repository clean and consistent, please follow these conventions when adding a new
notebook:

1. **Pick the correct domain folder** (or create a new one if your topic genuinely doesn't fit). Folder names use `lower_snake_case`.

2. **Name the file** `lower_snake_case.ipynb`, describing the *physics* and the *method* if relevant — e.g. `kdv_equation_split_step.ipynb`, not `notebook5.ipynb` or `untitled.ipynb`.

3. **Open every notebook with a tutorial-style markdown cell** containing the four sections:

   ```markdown
   # Title — Physical System and Method

   ## Physical system
   One paragraph describing what is being modelled.

   ## Governing equations
   The PDE / ODE / Hamiltonian in LaTeX. Use `$...$` for inline and `$$...$$` for display math.

   ## Numerical method
   The discretisation, time-stepper, or learning algorithm — and *why* this method was chosen.
   ```

4. **Insert short markdown cells before major plots** to explain what the figure shows and what to look for (the "Results & plots" part of the structure).

5. **Keep dependencies minimal**: prefer NumPy/SciPy/Matplotlib. Add anything new to `requirements.txt`.

6. **Reference data files with relative paths** from the notebook's location (e.g. `"../data/my_file.dat"` for a notebook one directory below `data/`).

7. **Update this README**: add a row to the relevant table in the *Notebook index*.

8. **Commit with a descriptive message** — e.g. `add KdV split-step solver to math_methods/`.

---

## License

This project is released under the MIT License — see [`LICENSE`](LICENSE) for details.
