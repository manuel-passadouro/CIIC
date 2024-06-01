import random
import csv

# Generate a list of unique random numbers from 1 to 100
unique_numbers = random.sample(range(1, 101), 3)

# Define the filename
filename = "Ecopoints.csv"

# Write the numbers to the CSV file
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(unique_numbers)

print(f"The list of unique random numbers has been saved to {filename}")
