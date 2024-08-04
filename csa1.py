import numpy as np
from math import gamma, pi

import numpy as np

n = 101  # Total number of nodes (100 customers + 1 depot)
k = 25   # Assuming 25 vehicles, adjust as needed

# X coordinates
x_i = [float(x) for x in [35, 41, 35, 55, 55, 15, 25, 20, 10, 55, 30, 20, 50, 30, 15, 30, 10, 5, 20, 15, 45, 45, 45, 55, 65, 65, 45, 35, 41, 64, 40, 31, 35, 53, 65, 63, 2, 20, 5, 60, 40, 42, 24, 23, 11, 6, 2, 8, 13, 6, 47, 40, 52, 45, 62, 60, 42, 16, 58, 34, 28, 35, 35, 25, 22, 22, 20, 20, 18, 15, 15, 30, 30, 28, 14, 25, 22, 8, 23, 4, 20, 20, 10, 10, 8, 8, 5, 5, 2, 0, 0, 36, 35, 33, 33, 32, 30, 34, 30, 36, 48, 26]]

# Y coordinates
y_i = [float(y) for y in [35, 49, 17, 45, 20, 30, 30, 50, 43, 60, 60, 65, 35, 25, 10, 5, 20, 30, 40, 60, 65, 20, 10, 5, 35, 20, 30, 40, 37, 42, 60, 52, 69, 52, 55, 65, 60, 20, 5, 12, 25, 7, 12, 3, 14, 38, 48, 56, 52, 68, 47, 50, 75, 70, 69, 66, 65, 42, 70, 60, 70, 66, 69, 85, 75, 85, 80, 85, 75, 75, 80, 50, 56, 52, 66, 50, 66, 62, 52, 55, 50, 55, 35, 40, 40, 45, 35, 45, 40, 40, 45, 18, 32, 32, 35, 20, 30, 25, 35, 40, 20, 32]]

# Earliest arrival times (ET)
ET = np.array([float(et) for et in [0, 0, 0, 0, 620, 0, 0, 0, 323, 329, 0, 146, 0, 639, 0, 118, 0, 0, 0, 0, 0, 0, 0, 146, 0, 716, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 234, 0, 0, 0, 0, 0, 167, 0, 0, 0, 0, 0, 0, 0, 213, 22, 1030, 1154, 15, 331, 965, 2653, 2385, 2628, 2603, 1985, 2310, 1846, 2077, 1763, 2143, 1560, 1689, 10, 2675, 12, 1519, 23, 2380, 1330, 93, 1268, 168, 170, 585, 448, 499, 1190, 666, 1076, 772, 890, 1036, 1612, 2599, 2529, 2463, 1745, 1932, 1884, 2348, 2715, 812, 2018]])

# Latest arrival times (LT)
LT = np.array([float(lt) for lt in [1000, 974, 972, 967, 860, 969, 978, 968, 563, 569, 964, 386, 975, 879, 957, 358, 960, 959, 974, 957, 958, 971, 963, 386, 960, 956, 978, 985, 983, 960, 964, 972, 956, 965, 953, 948, 948, 968, 474, 956, 978, 961, 964, 955, 407, 960, 954, 955, 962, 946, 973, 3390, 568, 563, 1463, 1527, 402, 822, 1340, 3280, 2976, 3113, 2952, 2412, 2659, 2365, 2514, 2264, 2638, 2083, 2144, 645, 3288, 505, 1926, 368, 2787, 1919, 484, 1785, 787, 595, 1136, 897, 1030, 1661, 1245, 1589, 1329, 1395, 1439, 2185, 3076, 2962, 2842, 2240, 2437, 2293, 2771, 3156, 1281, 2725]])

# Convert to numpy arrays
x_i = np.array(x_i)
y_i = np.array(y_i)

# Calculate distance matrix
distance_matrix = np.zeros((n, n))
for a in range(n):
    for b in range(n):
        distance = np.sqrt((x_i[a] - x_i[b])**2 + (y_i[a] - y_i[b])**2)
        distance_matrix[a][b] = distance

# 2D array of travel costs (C)
C = distance_matrix

# Penalty coefficient (now a single value)
a_i = 2.0


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

def CS(C, ET, LT, a_i, Pop_Size=50, MaxT=100, pa=0.25):
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
    best_solution, best_fitness = CS(C, ET, LT, a_i)
    print("\nBest Solution:", best_solution)
    print("Best Fitness:", best_fitness)

    print("\nInput Data:")
    print("Travel Costs (C):\n", C)
    print("Earliest Start Times (ET):", ET)
    print("Latest Service Times (LT):", LT)
    print("Penalty Coefficient (a_i):", a_i)
    
    if best_solution is not None:
        for i in range(k):
            if i < len(best_solution):
                route = [str(node) for node in [0] + best_solution[i] + [0]]
                print(f"Vehicle {i} = {' -> '.join(route)}")
            else:
                print(f"Vehicle {i} = No route (empty)")
    else:
        print("No valid solution found.")
    
    for i in range(k):
        route = [str(node) for node in [0] + best_solution[i] + [0]]
        print(f"Vehicle {i} = {' -> '.join(route)}")
        continue
