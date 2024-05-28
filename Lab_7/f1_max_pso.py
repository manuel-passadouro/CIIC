import random
import math

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

# Define the function f1(x1, x2)
def f1(individual):
    x1, x2 = individual
    z1 = math.sqrt(x1**2 + x2**2)
    z2 = math.sqrt((x1 - 1)**2 + (x2 + 1)**2)
    
    # To avoid division by zero
    if z1 == 0:
        term1 = 0
    else:
        term1 = math.sin(4 * z1) / z1
    
    if z2 == 0:
        term2 = 0
    else:
        term2 = math.sin(2.5 * z2) / z2
    
    return term1 + term2,

# Define the DEAP toolbox and the Genetic Algorithm
def main():
    # Define the creator
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Define the toolbox
    toolbox = base.Toolbox()
    
    # Attribute generator
    toolbox.register("attr_float", random.uniform, -5, 5)
    
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operator registering
    toolbox.register("evaluate", f1)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Genetic Algorithm parameters
    population_size = 300
    generations = 40
    cxpb = 0.7  # Crossover probability
    mutpb = 0.2  # Mutation probability

    # Initialize population
    population = toolbox.population(n=population_size)

    # Hall of Fame to store the best individual
    hof = tools.HallOfFame(1)

    # Statistics to collect during the run
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum([ind[0] for ind in x])/len(x))
    stats.register("std", lambda x: (sum([(ind[0]-sum([i[0] for i in x])/len(x))**2 for ind in x])/len(x))**0.5)
    stats.register("min", min)
    stats.register("max", max)

    # Run the Genetic Algorithm
    algorithms.eaSimple(population, toolbox, cxpb, mutpb, generations, stats=stats, halloffame=hof, verbose=True)

    # Print the best individual
    print("Best individual is:", hof[0], hof[0].fitness.values)

if __name__ == "__main__":
    main()
