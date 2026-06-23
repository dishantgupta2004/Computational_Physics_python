import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from flask import Flask, render_template, request
import streamlit as st

# Define the function to solve the differential equation using the Finite Element Method (FEM)
def finite_element_solver(a_func, c_func, q_func, L, u0, Q0, N):
    """
    Solve the differential equation -d/dx(a * du/dx) + c * u = q using finite element method.

    Parameters:
        a_func (function): Function defining coefficient a(x)
        c_func (function): Function defining coefficient c(x)
        q_func (function): Function defining source term q(x)
        L (float): Length of the domain (0 to L)
        u0 (float): Boundary condition at x=0 (u(0) = u0)
        Q0 (float): Flux boundary condition at x=L
        N (int): Number of elements

    Returns:
        x (numpy array): Grid points (nodes)
        u (numpy array): Solution at nodes
    """
    # Node coordinates and element size
    nodes = np.linspace(0, L, N + 1)
    h = L / N

    # Initialize global stiffness matrix and load vector
    K = np.zeros((N + 1, N + 1))
    F = np.zeros(N + 1)

    # Assemble the global stiffness matrix and load vector
    for e in range(N):
        x1, x2 = nodes[e], nodes[e + 1]
        a = lambda x: a_func((x1 + x2) / 2)
        c = lambda x: c_func((x1 + x2) / 2)
        q = lambda x: q_func((x1 + x2) / 2)

        # Local stiffness matrix and load vector
        Ke = np.array([
            [a(x1) / h + c(x1) * h / 3, -a(x1) / h + c(x1) * h / 6],
            [-a(x1) / h + c(x1) * h / 6, a(x1) / h + c(x1) * h / 3]
        ])
        Fe = np.array([
            q(x1) * h / 2,
            q(x2) * h / 2
        ])

        # Assemble into global matrix and vector
        K[e:e + 2, e:e + 2] += Ke
        F[e:e + 2] += Fe

    # Apply boundary conditions
    K[0, :] = 0
    K[0, 0] = 1
    F[0] = u0

    K[-1, -2] = -1 / h
    K[-1, -1] = 1 / h
    F[-1] = Q0

    # Solve the system of equations
    u = solve(K, F)

    return nodes, u


# Streamlit implementation
def streamlit_app():
    st.title("Finite Element Solver for Differential Equation")

    st.sidebar.header("Input Parameters")

    a_expr = st.sidebar.text_input("Coefficient a(x) (as a function of x)", value="1.0")
    c_expr = st.sidebar.text_input("Coefficient c(x) (as a function of x)", value="1.0")
    q_expr = st.sidebar.text_input("Source term q(x) (as a function of x)", value="1.0")
    L = st.sidebar.number_input("Domain length L", value=1.0)
    u0 = st.sidebar.number_input("Boundary condition u(0)", value=0.0)
    Q0 = st.sidebar.number_input("Flux boundary condition Q0", value=0.0)
    N = st.sidebar.slider("Number of elements N", min_value=2, max_value=100, value=10)

    if st.sidebar.button("Solve"):
        try:
            # Convert expressions to functions
            a_func = lambda x: eval(a_expr)
            c_func = lambda x: eval(c_expr)
            q_func = lambda x: eval(q_expr)

            x, u = finite_element_solver(a_func, c_func, q_func, L, u0, Q0, N)

            st.subheader("Solution")
            st.write("Nodes and Solution:")
            st.dataframe({"x": x, "u": u})

            st.subheader("Plot")
            plt.figure()
            plt.plot(x, u, label="Solution u(x)", marker='o')
            plt.xlabel("x")
            plt.ylabel("u(x)")
            plt.legend()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error: {e}")
            
            
streamlit_app()
