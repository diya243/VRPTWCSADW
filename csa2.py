import pandas as pd
import numpy as np
import glob
import os
from math import gamma, pi

#___________________________________________________________________________________________
def load_dataset(file_path):
    df = pd.read_csv(file_path, sep=",", skipinitialspace=True)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Force numeric conversion
    for col in ["XCOORD.", "YCOORD.", "READY TIME", "DUE DATE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with invalid data
    df = df.dropna()

    n = len(df)        # total customers (including depot)
    k = max(1, (n-1) // 4)  # exclude depot, assign 1 vehicle per 4 customers


    x_i = df["XCOORD."].to_numpy(dtype=float)
    y_i = df["YCOORD."].to_numpy(dtype=float)
    ET  = df["READY TIME"].to_numpy(dtype=float)
    LT  = df["DUE DATE"].to_numpy(dtype=float)

    # Distance matrix
    distance_matrix = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            distance_matrix[a][b] = np.sqrt((x_i[a] - x_i[b])**2 + (y_i[a] - y_i[b])**2)

    return n, k, distance_matrix, ET, LT

#____________________________________________________________________________________________________


def penalty_function(S_i, ET_i, LT_i, a_i):
    if S_i < ET_i:
        return float(a_i * (ET_i - S_i))
    elif ET_i <= S_i <= LT_i:
        return 0.0
    else:  # S_i > LT_i
        return float('inf')

def fitness_function_Z(vehicle_matrix1, C, ET, LT, a_i):
    total_cost = 0.0
    for route in vehicle_matrix1:
        if len(route) < 2:  # Skip empty routes
            continue
        route = [0] + route + [0]  # Add depot at start and end
        route_cost = 0.0
        time = 0.0
        for i in range(len(route) - 1):
            from_node, to_node = route[i], route[i+1]
            route_cost += float(C[from_node][to_node])
            time += float(C[from_node][to_node])  # Assume travel time = cost for simplicity
            if to_node != 0:  # Not returning to depot
                route_cost += float(penalty_function(time, ET[to_node], LT[to_node], a_i))
        total_cost += route_cost
    return total_cost

def initial_Population(Pop_Size, n, K):
    population = []
    for _ in range(Pop_Size):
        vehicle_matrix1 = [[] for _ in range(K)]
        customers = list(range(1, n))
        np.random.shuffle(customers)
        # Ensure each vehicle gets at least one customer
        for i in range(K):
            if customers:
                vehicle_matrix1[i].append(customers.pop())
        for customer in customers:
            vehicle = np.random.randint(K)
            vehicle_matrix1[vehicle].append(customer)
        population.append(vehicle_matrix1)
    return population

def Levy_Flight(beta):
    sigma = (gamma(1 + beta) * np.sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma)
    v = np.random.normal(0, 1)
    step = u / abs(v) ** (1 / beta)
    return step

def CS(C, ET, LT, a_i, n, k, Pop_Size=50, MaxT=100, pa=0.25):
    K = k
    population = initial_Population(Pop_Size, n, K)
    fitness = np.array([fitness_function_Z(nest, C, ET, LT, a_i) for nest in population], dtype=float)
    best_solution = population[0]
    best_fitness = np.inf
    
    for i in range(MaxT):
        new_population = []
        for nest in population:
            new_nest = nest.copy()
            # Perform a simple swap operation as Levy flight is not directly applicable
            if np.random.random() < 0.5:
                vehicle1, vehicle2 = np.random.choice(K, 2, replace=False)
                if len(new_nest[vehicle1]) > 0 and len(new_nest[vehicle2]) > 0:
                    cust1 = np.random.choice(new_nest[vehicle1])
                    cust2 = np.random.choice(new_nest[vehicle2])
                    new_nest[vehicle1].remove(cust1)
                    new_nest[vehicle2].remove(cust2)
                    new_nest[vehicle1].append(cust2)
                    new_nest[vehicle2].append(cust1)
            new_population.append(new_nest)
        
        new_fitness = np.array([fitness_function_Z(nest, C, ET, LT, a_i) for nest in new_population], dtype=float)
        
        replace_soln = np.where(new_fitness < fitness)[0]
        for idx in replace_soln:
            population[idx] = new_population[idx]
            fitness[idx] = new_fitness[idx]
        
        sorted_indices = np.argsort(fitness)
        population = [population[i] for i in sorted_indices]
        fitness = fitness[sorted_indices]
        
        if fitness[0] < best_fitness:
            best_solution = population[0]
            best_fitness = fitness[0]
            
        abandon_egg = int(pa * Pop_Size)
        for _ in range(abandon_egg):
            idx = np.random.randint(Pop_Size)
            population[idx] = initial_Population(1, n, K)[0]
            fitness[idx] = fitness_function_Z(population[idx], C, ET, LT, a_i)
        
        print(f"Iteration {i+1}/{MaxT}: Best_Fitness = {best_fitness}")

    return best_solution, best_fitness

if __name__ == "__main__":
    # Loop through all CSV files in the current folder
    for file_path in glob.glob("*.csv"):
        print("\n==============================")
        print(f"Running Cuckoo Search on dataset: {os.path.basename(file_path)}")
        print("==============================")

        # Load dataset
        n, k, C, ET, LT = load_dataset(file_path)
        a_i = 2.0  # penalty coefficient

        # Run CS algorithm (your existing function)
        best_solution, best_fitness = CS(C, ET, LT, a_i, k)

        # Print summary results
        print("\nBest Fitness:", best_fitness)
        for i in range(k):
            if i < len(best_solution):
                route = [str(node) for node in [0] + best_solution[i] + [0]]
                print(f"Vehicle {i+1}: {' -> '.join(route)}")
            else:
                print(f"Vehicle {i+1}: No route")

