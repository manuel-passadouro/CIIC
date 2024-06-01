import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

# Load distance matrix from Excel file
distance_matrix = pd.read_excel("Project2_DistancesMatrix.xlsx", index_col=0).values

# Define the central point (index 0)
CENTRAL = 0

# Read EcoPoints from CSV file
ecopoints_df = pd.read_csv("Ecopoints.csv", header=None)
# Extract the first row as a list of EcoPoints
ecopoints = ecopoints_df.iloc[0].tolist()

print(ecopoints)
print()

# Ensure the list has at most 100 items
if len(ecopoints) > 100:
    raise ValueError(f"The file contains more than 100 EcoPoints. Found {len(ecopoints)} entries.")

# Ensure all ecopoints are valid indices for the distance matrix
if any(point < 0 or point >= len(distance_matrix) for point in ecopoints):
    raise ValueError("Some EcoPoints are out of valid range for the distance matrix")

print(ecopoints)

# Define the fitness function
def distance_evaluate(individual):
    total_distance = 0
    num_visited = 0
    
    # Start from the central point
    current_point = CENTRAL
    for next_point in individual:
        if num_visited >= 30 and next_point in [3, 43, 52, 53, 58, 69, 71, 72, 73, 74, 75, 76, 77, 78, 92]:
            total_distance += distance_matrix[current_point][next_point] * 1.4  # Apply 40% penalty
        else:
            total_distance += distance_matrix[current_point][next_point]
        current_point = next_point
        num_visited += 1
    
    # Return to the central point
    total_distance += distance_matrix[current_point][CENTRAL]
    return total_distance,

# Set up the Genetic Algorithm
def setup_ga():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(ecopoints)), len(ecopoints))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", distance_evaluate)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

def main():
    toolbox = setup_ga()
    
    # Initialize population
    population = toolbox.population(n=300)
    
    # Define statistics and hall of fame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    hof = tools.HallOfFame(1)
    
    # Run the Genetic Algorithm
    try:
        for gen in range(200):
            # Apply the genetic operators (crossover, mutation)
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
            
            # Evaluate the fitness of the offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace the population with the offspring
            population[:] = offspring
            
            # Update the hall of fame with the best individuals
            hof.update(population)
            
            # Gather all the fitnesses in one list and print the stats
            record = stats.compile(population)
            print(f"gen: {gen}, nevals: {len(invalid_ind)}, {record}")
    except IndexError as e:
        print("IndexError:", e)
        for ind in population:
            print("Individual:", ind)
            if any(gene < 0 or gene >= len(ecopoints) for gene in ind):
                print("Invalid individual:", ind)

    # Retrieve the best individual
    best_ind = hof[0]
    best_route = [CENTRAL] + best_ind + [CENTRAL]
    best_distance = distance_evaluate(best_ind)[0]
    
    # Output the result
    result = f"C, " + ", ".join([f"{distance_matrix[best_route[i]][best_route[i+1]]:.1f}E{best_route[i+1]:02d}" for i in range(len(best_route)-1)]) + f", Total={best_distance:.1f}"
    print("Best route:", result)

if __name__ == "__main__":
    main()
