import random
import numpy as np
import pandas as pd

# Load distance matrix from Excel file
distance_matrix = pd.read_excel("Project2_DistancesMatrix.xlsx", index_col=0).values

# Define the central point
CENTRAL = 0

# Read EcoPoints from CSV file
ecopoints_df = pd.read_csv("Ecopoints.csv", header=None)
# Extract the first row as a list of EcoPoints
ecopoints = ecopoints_df.iloc[0].tolist()

# Ensure the list has at most 100 items
if len(ecopoints) > 100:
    raise ValueError(f"The file contains more than 100 EcoPoints. Found {len(ecopoints)} entries.")

print("EcoPoints:", ecopoints)

# Define the fitness function
def evaluate(individual):
    total_distance = 0
    num_visited = 0
    
    # Start from the central point
    current_point = CENTRAL
    distances = []
    for i in range(len(individual)):
        next_point = individual[i]
        if num_visited >= 30 and next_point in [2, 42, 51, 52, 57, 68, 70, 71, 72, 73, 74, 75, 76, 77, 91]:  # adjusted for 0-indexed
            distance = distance_matrix[current_point][next_point] * 1.4  # Apply 40% penalty
        else:
            distance = distance_matrix[current_point][next_point]
        distances.append(distance)
        total_distance += distance
        current_point = next_point
        num_visited += 1
    
    # Return to the central point
    return_distance = distance_matrix[current_point][CENTRAL]
    distances.append(return_distance)
    total_distance += return_distance
    
    return distances, total_distance

# Generate a random path through the EcoPoints
random_path = random.sample(ecopoints, len(ecopoints))
print("Random Path:", random_path)

# Evaluate the random path
distances, total_distance = evaluate(random_path)
print("Distances between points:", distances)
print("Total distance:", total_distance)

