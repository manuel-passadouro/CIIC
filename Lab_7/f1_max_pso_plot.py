import operator
import random
import numpy
import math
import time
from deap import base, creator, tools
from colorama import init, Fore

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

# Create objects in the creator space
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, smin=None, smax=None, best=None)

# Define functions to generate and update particles
def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) # Random position
    part.speed = [random.uniform(smin, smax) for _ in range(size)] # Random speed
    part.smin = smin
    part.smax = smax
    return part

def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))

# Register operators in the toolbox
toolbox = base.Toolbox()
toolbox.register("particle", generate, size=2, pmin=-5, pmax=5, smin=-2, smax=2) #limit position and speed
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=1.5, phi2=2.5) #cognitive weight and social weight
toolbox.register("evaluate", f1)

def main():

    # Count the time
    start_time = time.time()

    pop = toolbox.population(n=100)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields
    
    best_fitnesses = []  # List to store the best fitness values of each generation
    best_x1_values = []  # List to store the x1 values of the best individuals
    best_x2_values = []  # List to store the x2 values of the best individuals

    GEN = 200
    
    # Create a Hall of Fame object to track the best individual
    hof = tools.HallOfFame(1)
    
    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not hof or hof[0].fitness < part.fitness:
                hof.update([part])
        for part in pop:
            toolbox.update(part, hof[0])

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
        
        # Append the best fitness value of the current generation to the list
        best_fitnesses.append(hof[0].fitness.values[0])
        # Append the x1 and x2 values of the best individual to the respective lists
        best_x1_values.append(hof[0][0])
        best_x2_values.append(hof[0][1])
    
    # Print the best individual found
    print("Best individual:", hof[0])

    end_time  = time.time()
    total_time = end_time - start_time
    print()
    print(f"{Fore.YELLOW}Algorithm execution time:")
    print()
    print(f"{Fore.BLUE}{total_time:.2f} seconds")
    print()
    
    # Plot the evolution of the best fitness, x1, and x2 together with each generation
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, GEN+1), best_x1_values, label='x1')
    plt.plot(range(1, GEN+1), best_x2_values, label='x2')
    plt.plot(range(1, GEN+1), best_fitnesses, label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.title('Evolution of Best Fitness, x1, and x2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'evolution_PSO.png')

    return pop, logbook, hof[0]

if __name__ == "__main__":
    main()

