import numpy as np
import pyomo.environ as pyo

def solve_optimization(anchors, measured_distances):
    # Horizon length for the optimization problem
    horizon_length = len(measured_distances)

    # Define the model
    model = pyo.ConcreteModel()

    # Define the variables
    model.x = pyo.Var(range(2))  # User position variables x and y

    # Define the objective function
    def squared_error_rule(model):
        return sum((measured_distances[i] - sum((model.x[0] - anchors[j][0])**2 + (model.x[1] - anchors[j][1])**2 for j in range(len(anchors))))**2 for i in range(horizon_length))
    model.objective = pyo.Objective(rule=squared_error_rule, sense=pyo.minimize)

    # Solve the optimization problem
    solver = pyo.SolverFactory('ipopt')
    solver.solve(model)

    # Extract the optimized user position
    optimized_user_position = np.array([model.x[0].value, model.x[1].value])

    return optimized_user_position

# Anchor positions
anchors = np.array([[100, 200], [400, 400], [700, 200]])

# Load measured distances from CSV
measured_distances_data = np.genfromtxt('distances.csv', delimiter=',', skip_header=1)
time_points = measured_distances_data[:, 0]
measured_distances = measured_distances_data[:, 1:]

# Optimization for each time point
optimized_positions = []
for i, time in enumerate(time_points):
    optimized_position = solve_optimization(anchors, measured_distances[i])
    optimized_positions.append(optimized_position)
    print(f"Time: {time}, Optimized user position: {optimized_position}")

# Write-up of results
for i, time in enumerate(time_points):
    print(f"At time {time}, the optimized position of the user is {optimized_positions[i]}")
