import random
import math
import operator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from deap import base, creator, tools, algorithms

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

#Define functions to generate and update particles.
def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) #Random position.
    part.speed = [random.uniform(smin, smax) for _ in range(size)] #Random speed.
    part.smin = smin
    part.smax = smax
    return part


# Define the DEAP toolbox and the Particle Swarm Optimization
def main():
    # Define the creator
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, smin=None, smax=None, best=None)

    # Define the toolbox
    toolbox = base.Toolbox()
    
    # Attribute generator
    toolbox.register("attr_float", random.uniform, -5, 5)
    
    # Structure initializers
    toolbox.register("particle", generate, size=2, pmin=-6, pmax=6, smin=-3, smax=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)

    # Operator registering
    toolbox.register("evaluate", f1)
    toolbox.register("update", updateParticle)
    toolbox.register("inertia", random.uniform, 0.5, 1.0)

    # Parameters for PSO
    population_size = 200
    generations = 100
    phi1 = 3.5  # Cognitive parameter
    phi2 = 2.0  # Social parameter

    # Initialize population
    population = toolbox.population(n=population_size)
    
    # Set speed limits
    for part in population:
        part.speed = [random.uniform(-3, 3) for _ in range(len(part))]
        part.smin = -2
        part.smax = 2

    # Statistics to collect during the run
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum([ind[0] for ind in x])/len(x))
    stats.register("std", lambda x: (sum([(ind[0]-sum([i[0] for i in x])/len(x))**2 for ind in x])/len(x))**0.5)
    stats.register("min", min)
    stats.register("max", max)

    # Lists to store the evolution of x1, x2, and max f1 for each generation
    x1_evolution = []
    x2_evolution = []
    f1_max_evolution = []

    # Define a function to log the evolution data
    def logbook_record(pop):
        best_ind = tools.selBest(pop, 1)[0]
        x1, x2 = best_ind
        f1_value = best_ind.fitness.values[0]
        x1_evolution.append(x1)
        x2_evolution.append(x2)
        f1_max_evolution.append(f1_value)

    # Run the Particle Swarm Optimization
    for gen in range(generations):
        for part in population:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values

        best = tools.selBest(population, 1)[0]
        for part in population:
            toolbox.update(part, best, phi1, phi2)

        logbook_record(population)

    # Print the best individual
    best_ind = tools.selBest(population, 1)[0]
    print("Best individual is:", best_ind, best_ind.fitness.values)

    # Plot the evolution of x1, x2, and max f1 on the same plot
    generations_range = range(generations)

    plt.figure(figsize=(10, 6))

    plt.plot(generations_range, x1_evolution, label='x1')
    plt.plot(generations_range, x2_evolution, label='x2')
    plt.plot(generations_range, f1_max_evolution, label='max f1', linestyle='--')
    
    plt.xlabel('Generation')
    plt.ylabel('Values')
    plt.title('Evolution of x1, x2, and max f1')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'evolution_PSO.png')

    # 3D plot of f1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x, y)
    z = np.vectorize(lambda x, y: f1([x, y])[0])(x, y)

    angles = [(30, 30), (60, 30), (90, 30), (30,60), (30,90)]

    for i, (elev, azim) in enumerate(angles):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, alpha=0.75, cmap='viridis')

        # Mark the best individual found by the PSO
        best_x1, best_x2 = best_ind
        best_z = f1(best_ind)[0]
        ax.scatter(best_x1, best_x2, best_z, color='r', s=100)  # Red color, size 100

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('f1(X1, X2)')

        ax.view_init(elev=elev, azim=azim)
        plt.savefig(f'f1_PSO_3D_{elev}_{azim}.png')
        plt.close(fig)
        print(f"Plot saved as 'f1_PSO_3D_{elev}_{azim}.png'.")

if __name__ == "__main__":
    main()

