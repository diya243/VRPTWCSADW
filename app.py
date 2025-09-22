import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from csa2 import load_dataset, CS

# ------------------------------------------------------------------
# Helper: Plot routes on 2D graph
def plot_routes(best_solution, x_coords, y_coords):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.get_cmap("tab20", len(best_solution))

    # Plot depot
    ax.scatter(x_coords[0], y_coords[0], c="red", s=200, marker="s", label="Depot")

    for idx, route in enumerate(best_solution):
        if not route:
            continue
        route_nodes = [0] + route + [0]
        xs = [x_coords[node] for node in route_nodes]
        ys = [y_coords[node] for node in route_nodes]
        ax.plot(xs, ys, marker="o", color=colors(idx), label=f"Vehicle {idx+1}")

    ax.set_title("Vehicle Routes")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    return fig

# ------------------------------------------------------------------
st.title("🚚 VRPTW Solver - Cuckoo Search")

uploaded_file = st.file_uploader("Upload a dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    file_path = os.path.join("temp.csv")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load dataset
    n, _, C, ET, LT = load_dataset(file_path)
    df = pd.read_csv(file_path)
    st.write("### Dataset Preview", df.head())

    # Automatically calculate number of vehicles
    k = max(1, (n - 1) // 4)  # 1 vehicle per 4 customers (excluding depot)
    st.info(f"Number of vehicles (k) determined for this dataset: {k}")

    # Penalty coefficient
    a_i = 2.0

    # Run CS algorithm
    st.write("### Running Cuckoo Search Algorithm...")
    best_solution, best_fitness = CS(C, ET, LT, a_i, n=n,  k=k, Pop_Size=50, MaxT=100, pa=0.25)

    # Display results
    st.success(f"✅ Best Fitness (Objective Value): {best_fitness:.2f}")

    # Metrics
    total_distance = best_fitness  # fitness includes distance + penalties
    avg_route_length = total_distance / max(1, len([r for r in best_solution if r]))
    st.metric("Total Distance", f"{total_distance:.2f}")
    st.metric("Average Route Length", f"{avg_route_length:.2f}")

    # Display routes
    st.write("### Vehicle Routes")
    for i, route in enumerate(best_solution):
        if route:
            st.write(f"Vehicle {i+1}: " + " → ".join(str(node) for node in [0] + route + [0]))
        else:
            st.write(f"Vehicle {i+1}: No route")

    # Plot routes
    st.write("### Route Visualization")
    x_coords = df["XCOORD."].to_numpy(dtype=float)
    y_coords = df["YCOORD."].to_numpy(dtype=float)
    fig = plot_routes(best_solution, x_coords, y_coords)
    st.pyplot(fig)

