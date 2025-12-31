import numpy as np
import scipy.linalg
import streamlit as st
import sympy as sp

# Streamlit Interface
st.title("Finite Element Method Solver for General Cases")
st.sidebar.header("Problem Setup")
st.sidebar.write(
    "Input the coefficients \(a(x)\), \(c(x)\), and \(q(x)\) as functions of \(x\), boundary conditions, and domain details."
)

# Inputs for domain
L = st.sidebar.number_input("Domain Length (L)", min_value=0.1, value=1.0)
N = st.sidebar.number_input("Number of Elements (N)", min_value=2, value=10, step=1)
h = L / N

# Symbolic variable for defining a(x), c(x), q(x)
x = sp.Symbol("x")

# Coefficients as functions of x
a_expr = st.sidebar.text_input("Diffusion Coefficient (a(x))", value="1")
c_expr = st.sidebar.text_input("Reaction Coefficient (c(x))", value="0")
q_expr = st.sidebar.text_input("Source Term (q(x))", value="1")

# Convert to callable functions
a_func = sp.lambdify(x, sp.sympify(a_expr))
c_func = sp.lambdify(x, sp.sympify(c_expr))
q_func = sp.lambdify(x, sp.sympify(q_expr))

# Boundary conditions
u_0 = st.sidebar.number_input("Boundary Condition at x=0 (u(0))", value=0.0)
Q_0 = st.sidebar.number_input("Flux Condition at x=L ((a * du/dx)(L))", value=0.0)

# Element and Node information
nodes = np.linspace(0, L, N + 1)
elements = [(nodes[i], nodes[i + 1]) for i in range(N)]

# Global stiffness matrix and load vector
K = np.zeros((N + 1, N + 1))
F = np.zeros(N + 1)

# Assemble the stiffness matrix and load vector
for e, (x0, x1) in enumerate(elements):
    h_e = x1 - x0
    
    # Midpoint for approximating coefficients
    x_mid = (x0 + x1) / 2
    a_mid = a_func(x_mid)
    c_mid = c_func(x_mid)
    q_mid = q_func(x_mid)

    # Local stiffness matrix (2x2)
    k_local = (
        a_mid / h_e * np.array([[1, -1], [-1, 1]]) +
        c_mid * h_e / 6 * np.array([[2, 1], [1, 2]])
    )

    # Local load vector (2x1)
    f_local = q_mid * h_e / 2 * np.array([1, 1])

    # Assemble into global matrix/vector
    K[e:e + 2, e:e + 2] += k_local
    F[e:e + 2] += f_local

# Apply boundary conditions
K[0, :] = 0
K[0, 0] = 1
F[0] = u_0

K[-1, :] = 0
K[-1, -1] = 1
F[-1] += Q_0 / a_func(L)  # Neumann condition at x=L

# Solve the system
u = scipy.linalg.solve(K, F)

# Results
st.header("Results")
st.write(f"Domain discretized into {N} elements.")

# Plot the solution
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(nodes, u, marker="o", label="FEM Solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Solution of the Differential Equation")
plt.grid(True)
plt.legend()
st.pyplot(plt)

# Display results in a table
st.subheader("Nodal Values")
for i, value in enumerate(u):
    st.write(f"Node {i}: u({nodes[i]:.2f}) = {value:.4f}")

# Input descriptions
st.sidebar.subheader("Descriptions of Inputs")
st.sidebar.markdown(
    "**Domain Length (L):** The length of the 1D domain where the problem is solved.\n\n"
    "**Number of Elements (N):** Number of finite elements the domain is divided into.\n\n"
    "**Diffusion Coefficient (a(x)):** Represents the strength of the diffusion term. Example: 1, x, sin(x).\n\n"
    "**Reaction Coefficient (c(x)):** Represents the reaction term's strength. Example: 0, x^2, exp(x).\n\n"
    "**Source Term (q(x)):** Represents the source term in the differential equation. Example: 1, sin(x).\n\n"
    "**Boundary Condition (u(0)):** Value of the solution at the left boundary (essential boundary condition).\n\n"
    "**Flux Condition ((a * du/dx)(L)):** Flux value at the right boundary (natural boundary condition)."
)
