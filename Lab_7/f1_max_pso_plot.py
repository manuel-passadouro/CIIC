import random
import operator
import math

import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools

# Define the function f1(x1, x2)
def f1(individual):
    x1, x2 = individual
    z1 = math.sqrt(x1**2 + x2**2)
    z2 = math.sqrt((x1 - 1)**2 + (x2 + 1)**2)
    
    # To avoid division by zero
    term1 = math.sin(4 * z1) / z1 if z1 != 0 else 0
    term2 = math.sin(2.5 * z2) / z2 if z2 != 0 else 0
    
    return term1 + term2,

# Set up DEAP components
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, best=None, bestfit=None)
creator.create("Swarm", list, best=None, bestfit=None)

# Function to generate particles
def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle([random.uniform(pmin, pmax) for _ in range(size)])
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

# Function to update particles
def updateParticle(part, best, phi1, phi2):
    u1 = [random.uniform(0, phi1) for _ in range(len(part))]
    u2 = [random.uniform(0, phi2) for _ in range(len(part))]
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=2, pmin=-5, pmax=5, smin=-1, smax=1)
toolbox.register("swarm", tools.initRepeat, creator.Swarm, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", f1)

def main():
    # Initialize swarm
    swarm = toolbox.swarm(n=100)
    
    # Evaluate initial fitness
    for part in swarm:
        part.fitness.values = toolbox.evaluate(part)
        part.best = part[:]
        part.bestfit = part.fitness.values
    
    # Initialize the best
    best = None
    
    # PSO parameters
    generations = 50
    
    # Prepare for plotting
    fig, ax = plt.subplots()
    plt.ion()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    scatter = ax.scatter([], [], c='blue', label='Particles')
    best_scatter = ax.scatter([], [], c='red', label='Best Particle')
    ax.legend()
    
    for gen in range(generations):
        for part in swarm:
            part.fitness.values = toolbox.evaluate(part)
            if part.fitness.values > part.bestfit:
                part.best = part[:]
                part.bestfit = part.fitness.values
            if best is None or part.fitness.values > best.bestfit:
                best = part
                
        for part in swarm:
            toolbox.update(part, best)
        
        # Plotting the particles
        swarm_positions = np.array([part for part in swarm])
        scatter.set_offsets(swarm_positions)
        
        # Plotting the best particle
        best_position = np.array(best)
        best_scatter.set_offsets([best_position])
        
        plt.draw()
        plt.pause(0.1)
        
        print(f"Generation {gen}, Best Fitness: {best.bestfit[0]}")
    
    print("Best individual is:", best, best.fitness.values)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
