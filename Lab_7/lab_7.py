import array
import numpy
import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

# Define the Fitness and Individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Single objective fitness
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create types in toolbox container
toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    return sum(individual),

# Register operators in the toolbox
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Evolutionary loop
    for gen in range(40):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the population with the offspring
        pop[:] = offspring

        # Update the hall of fame with the current generation
        hof.update(pop)
        
        # Update the statistics
        record = stats.compile(pop)
        print(record)
        
        # Find the best individual and its index
        best_ind = tools.selBest(pop, 1)[0]
        best_index = pop.index(best_ind)
        print(f"Generation {gen}: Best individual index = {best_index}, Fitness = {best_ind.fitness.values[0]}")

    return pop, stats, hof

if __name__ == "__main__":
    main()

