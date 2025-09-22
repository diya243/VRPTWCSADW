import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from csa2 import load_dataset, CS

# ------------------------------------------------------------------
def plot_routes(best_solution, x_coords, y_coords):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.get_cmap("tab20", len(best_solution))
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

# Sidebar for dataset selection
st.sidebar.header("Choose Dataset")
source_dir = "."  # same directory as app.py
sample_files = [f for f in os.listdir(source_dir) if f.endswith(".csv")][14:19]  # first 5 CSVs

dataset_option = st.sidebar.radio(
    "Select a dataset",
    ["Upload my own CSV"] + sample_files
)

if dataset_option == "Upload my own CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        file_path = os.path.join("temp.csv")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        st.stop()
else:
    file_path = os.path.join(source_dir, dataset_option)

# ------------------------------------------------------------------
n, _, C, ET, LT = load_dataset(file_path)
df = pd.read_csv(file_path)
st.write("### Dataset Preview", df.head())

k = max(1, (n - 1) // 4)
st.info(f"Number of vehicles (k) determined for this dataset: {k}")
a_i = 2.0

pop_size = st.sidebar.number_input("Population Size", min_value=10, max_value=200, value=50, step=10)
max_iter = st.sidebar.number_input("Max Iterations", min_value=10, max_value=500, value=100, step=10)
pa = st.sidebar.slider("Abandonment Probability (pa)", 0.0, 1.0, 0.25, 0.05)

st.write("### Running Cuckoo Search Algorithm...")
best_solution, best_fitness, fitness_history = CS(C, ET, LT, a_i, n=n, k=k, Pop_Size=pop_size, MaxT=max_iter, pa=pa)

st.success(f"✅ Best Fitness (Objective Value): {best_fitness:.2f}")

total_distance = best_fitness
avg_route_length = total_distance / max(1, len([r for r in best_solution if r]))
st.metric("Total Distance", f"{total_distance:.2f}")
st.metric("Average Route Length", f"{avg_route_length:.2f}")

st.write("### Vehicle Routes")
for i, route in enumerate(best_solution):
    if route:
        st.write(f"Vehicle {i+1}: " + " → ".join(str(node) for node in [0] + route + [0]))
    else:
        st.write(f"Vehicle {i+1}: No route")

# Route plot
st.write("### Route Visualization")
x_coords = df["XCOORD."].to_numpy(dtype=float)
y_coords = df["YCOORD."].to_numpy(dtype=float)
fig = plot_routes(best_solution, x_coords, y_coords)
st.pyplot(fig)

# Fitness progression plot
st.write("### Fitness Progression Over Iterations")
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(fitness_history, marker="o")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Best Fitness")
ax2.set_title("Fitness Improvement During Cuckoo Search")
st.pyplot(fig2)
