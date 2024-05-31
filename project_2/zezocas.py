import random
import time
import pandas as pd
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools

# Pandas file
distances_df = pd.read_excel("Project2_DistancesMatrix.xlsx", index_col=0)
# mudar para índices
distances = distances_df.copy()
distances = distances.reset_index(drop=True)
distances.columns = range(0, len(distances.columns))
N_POINTS_COLUMNS = len(distances.columns)
N_POINTS_ROWS = len(distances)
N_POINTS = N_POINTS_ROWS
M_POINT_IN = N_POINTS - 1
#print(len(distances))
#print("N points:",N_POINTS)
#print(distances)

# Functions
# Evaluation function
def evalTSP(individual):
    distance_calc = distances[0][M_POINT_IN]
    for i in range(1,N_POINTS):
        distance_calc = distance_calc + distances[individual[i-1]][individual[i]]

    return distance_calc,

# Code
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Attribute generator 
toolbox.register("attr_int", random.sample, range(N_POINTS), N_POINTS)
# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_int)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalTSP)
toolbox.register("mate", tools.cxOrdered) #the order of elements is critical
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05) # indpb = 0.05
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=300)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # CXPB is the probability with which two individuals
    # are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2
    
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    N_GENS = 500

    # Hall of Fame and vec for figure
    hof = tools.HallOfFame(maxsize=5)
    min_v = [0] * N_GENS
    max_v = [0] * N_GENS
    mean_v = [0] * N_GENS

    # Count the time
    start_time = time.time()
    # Begin the evolution
    while g < N_GENS: # max(fits) < 100 and g < 100
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
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
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        min_v[g-1] = min(fits)
        max_v[g-1] = max(fits)
        mean_v[g-1] = mean

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        min_id = fits.index(min(fits))
        #print(min_id)
        print("Best sequence from this generation:",pop[min_id])

        # Update HoF
        hof.update(pop)

    end_time  = time.time()
    algo_time = end_time - start_time
    print("Algorithm execution time:", algo_time)

    print("Best 5 individuals\n")
    converted_hof = []
    rotated_sequence = []
    mapping = {num: f'E{num}' if num > 0 else 'C' for num in range(100)}
    for i in range(len(hof)):
        sequence = [mapping[num] for num in hof[i]]
        c_index = sequence.index("C")
        rotated_sequence = sequence[c_index:] + sequence[:c_index]
        converted_hof.append(rotated_sequence)
        
    for i in range(len(hof)):
        print("Sequence", i, ": Distance", evalTSP(hof[i]))
        #print(hof[i])
        print(converted_hof[i])

    plt.plot(range(1,N_GENS+1),min_v, label = "Min")
    plt.plot(range(1,N_GENS+1),max_v, label = "Max")
    plt.plot(range(1,N_GENS+1),mean_v, label = "Mean")
    plt.ylabel("Distance")
    plt.xlabel("Generation")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()