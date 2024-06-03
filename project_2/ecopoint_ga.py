import time
import random
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from deap import base, creator, tools, algorithms
import matplotlib
matplotlib.use('tkagg')  # For interactive graphics in wsl
import matplotlib.pyplot as plt
from colorama import init, Fore

# Initialize colorama
init(autoreset=True)

# Define the central and penalty points
CENTRAL = 0
PENALITY = [3, 43, 52, 53, 58, 69, 71, 72, 73, 74, 75, 76, 77, 78, 92]

# Define the size of population and the number of generations
POP_SIZE = 1900
NUM_GEN = 1000

# Load distance matrix from Excel file
distance_matrix = pd.read_excel("Project2_DistancesMatrix.xlsx", index_col=0).values

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Ecopoints List File")
    parser.add_argument("ecopoints_file", help="Input CSV file containing the Ecopoints index")
    return parser.parse_args()

# Define the fitness function
def evaluate(individual, distance_matrix, ecopoints_mapping):
    total_distance = 0
    num_visited = 0
    
    # Start from the central point
    current_point = CENTRAL

    for i in range(len(individual)):
        # Map the individual point from standard to original
        next_point = ecopoints_mapping[individual[i]]
        
        if num_visited >= 30 and next_point in PENALITY:
            total_distance += distance_matrix[current_point][next_point] * 1.4  # Apply 40% penalty
        else:
            total_distance += distance_matrix[current_point][next_point]
        current_point = next_point
        num_visited += 1
    
    # Return to the central point
    total_distance += distance_matrix[current_point][CENTRAL]
    return total_distance,

# Set up the Genetic Algorithm
def setup_ga(distance_matrix, ecopoints_mapping, ecopoints_standard):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(ecopoints_standard)), len(ecopoints_standard))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", lambda ind: evaluate(ind, distance_matrix, ecopoints_mapping))
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

# Function to get adjusted distance
def get_distance(current_point, next_point, tracker):
    distance = distance_matrix[current_point][next_point]
    if (tracker > 30) and (next_point in PENALITY):
        distance *= 1.4  # Increase distance by 40%
    return distance

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Read EcoPoints from CSV file
    try:
        ecopoints_df = pd.read_csv(args.ecopoints_file, header=None)
    except Exception as e:
        print(f"Error reading EcoPoints file: {e}")
        return
    
    # Extract the first row as a list of EcoPoints
    ecopoints = ecopoints_df.iloc[0].tolist()

    # Ensure the list has at most 100 items
    if len(ecopoints) > 100:
        raise ValueError(f"The file contains more than 100 EcoPoints. Found {len(ecopoints)} entries.")

    # Create the mapping and standard lists
    ecopoints_standard = list(range(len(ecopoints)))
    ecopoints_mapping = {i: point for i, point in enumerate(ecopoints)}

    toolbox = setup_ga(distance_matrix, ecopoints_mapping, ecopoints_standard)

    # Initialize population
    population = toolbox.population(n=POP_SIZE)
    
    # Define statistics and hall of fame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Save the best individual
    hof = tools.HallOfFame(1)
    
    # Count the time
    start_time = time.time()

    # Run the Genetic Algorithm
    logbook = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=NUM_GEN, stats=stats, halloffame=hof, verbose=False)

    # Extract data from logbook
    gen = []
    nevals = []
    avg = []
    std = []
    min_ = []
    max_ = []

    for entry in logbook[1]:
        gen.append(entry['gen'])
        nevals.append(entry['nevals'])
        avg.append(entry['avg'])
        std.append(entry['std'])
        min_.append(entry['min'])
        max_.append(entry['max'])
        
    logbook_df = pd.DataFrame({
        'gen': gen,
        'nevals': nevals,
        'avg': avg,
        'std': std,
        'min': min_,
        'max': max_
    })
    logbook_df.to_csv("Logbook.csv", index=False)
    #print(logbook_df)

    end_time  = time.time()
    total_time = end_time - start_time
    print()
    print(f"{Fore.YELLOW}Algorithm execution time:")
    print()
    print(f"{Fore.BLUE}{total_time:.2f} seconds")
    print()

    # Retrieve and print the best individual from the Hall of Fame
    best_ind_standard = hof[0]
    best_ind_original = [ecopoints_mapping[point] for point in best_ind_standard]
    best_route = [CENTRAL] + best_ind_original + [CENTRAL]
    best_distance = evaluate(best_ind_standard, distance_matrix, ecopoints_mapping)[0]

    # Output the result, add starting "C" manually, 
    # if node is 0 then it's back at the central (should always be the last node).
    result = "C, " + ", ".join([
        (Fore.RED if (best_route[i+1] in PENALITY and i >= 30) else Fore.WHITE) +
        f"{get_distance(best_route[i], best_route[i+1], i):.1f}C" if best_route[i+1] == 0 
        else (Fore.RED if (best_route[i+1] in PENALITY and i >= 30) else Fore.WHITE) +
        f"{get_distance(best_route[i], best_route[i+1], i):.1f}E{best_route[i+1]:02d}" 
        for i in range(len(best_route)-1)
    ]) + f", {Fore.BLUE}Total={best_distance:.2f}"

    print(f"{Fore.YELLOW}Best route:")
    print()
    print(result)
    print()

    # Re-open the Logbook.csv and extract values from the second line
    logbook_df = pd.read_csv("Logbook.csv")

    gen = logbook_df['gen'].tolist()
    nevals = logbook_df['nevals'].tolist()  # Not being used...
    avg = logbook_df['avg'].tolist()
    std = logbook_df['std'].tolist()  # Not being used...
    min_ = logbook_df['min'].tolist()
    max_ = logbook_df['max'].tolist()

    # Plot statistics
    plt.figure()
    plt.plot(gen, min_, label="Min")
    plt.plot(gen, max_, label="Max")
    plt.plot(gen, avg, label="Avg")
    plt.xlabel("Generation")
    plt.ylabel("Distance (km)")
    plt.legend()
    plt.savefig("max_avg_min.png")  # Save the plot as a PNG file
    plt.close()  # Close the plot to prevent it from being displayed

    # Create a directed graph to visualize the best route
    G = nx.DiGraph()

    for i in range(len(best_route)-1):
        G.add_edge(best_route[i], best_route[i+1], weight=distance_matrix[best_route[i]][best_route[i+1]])

    # Draw the graph
    plt.figure(figsize=(16, 16))
    pos = nx.circular_layout(G)
    labels = {node: ('C' if node == CENTRAL else str(node)) for node in G.nodes()}  # Replace 0 with 'C' for the central node
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1000)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='black', width=1.5)  # Use lines instead of arrows
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), font_color='black')
    nx.draw_networkx_nodes(G, pos, nodelist=[CENTRAL], node_color='red', node_size=1000)  # Paint central node red
    plt.title('Best Route', fontsize=20, fontweight='bold')  # Bold title
    plt.savefig("best_route.png", format="png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
