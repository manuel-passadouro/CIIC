import random
import numpy as np
import pandas as pd
from deap import base, creator, tools

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

# Define the fitness function
def evaluate(individual):
    total_distance = 0
    num_visited = 0
    
    # Start from the central point
    current_point = CENTRAL
    for point in individual:
        if num_visited >= 30 and point in [2, 42, 51, 52, 57, 68, 70, 71, 72, 73, 74, 75, 76, 77, 91]:  # adjusted for 0-indexed
            total_distance += distance_matrix[current_point][point] * 1.4  # Apply 40% penalty
        else:
            total_distance += distance_matrix[current_point][point]
        current_point = point
        num_visited += 1
    
    # Return to the central point
    total_distance += distance_matrix[current_point][CENTRAL]
    
    return total_distance,

# Create the DEAP fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Create the DEAP toolbox
toolbox = base.Toolbox()

# Register functions to create individuals and populations
toolbox.register("indices", random.sample, range(1, len(ecopoints) + 1), len(ecopoints))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the evaluation function
toolbox.register("evaluate", evaluate)

# Define the ordered crossover function for TSP
def cxOrdered(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    temp1, temp2 = ind1[cxpoint1:cxpoint2+1], ind2[cxpoint1:cxpoint2+1]
    ind1 = [x for x in ind1 if x not in temp2]
    ind2 = [x for x in ind2 if x not in temp1]
    ind1[cxpoint1:cxpoint2+1], ind2[cxpoint1:cxpoint2+1] = temp2, temp1
    return ind1, ind2

# Register the ordered crossover function in DEAP toolbox
toolbox.register("mate", cxOrdered)

# Register genetic operators
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(42)
    
    # Create an initial population of 100 individuals
    population = toolbox.population(n=100)
    
    # Crossover and mutation probabilities
    CXPB, MUTPB = 0.7, 0.2
    
    # Evaluate the entire population
    fitnesses = [toolbox.evaluate(ind) for ind in population]
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Extract fitness values
    fits = [ind.fitness.values[0] for ind in population]
    
    # Begin the evolution
    g = 0
    while g < 100:
        g += 1
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [toolbox.evaluate(ind) for ind in invalid_ind]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace the population by the offspring
        population[:] = offspring
        
        # Gather all the fitnesses in this generation
        fits = [ind.fitness.values[0] for ind in population]
        
        # Print statistics
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = np.sqrt((sum2 / length) - (mean * mean))
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(population, 1)[0]
    print("Best individual:", best_ind)
    print("Best fitness:", best_ind.fitness.values[0])

if __name__ == "__main__":
    main()