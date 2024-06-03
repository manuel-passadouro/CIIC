To run the code, the user only needs to call the name of the file and pass the file containing the list of ecopoints as an argument:

>> python3 ecopoint_ga.py ECOPOINTS.csv

PS:. The code is prepared to receive an argument file with any name.

To successfully run the code, it is necessary to install the following packages:

>> pip3 install networkx
>> pip3 install colorama

PS: The first package is required to generate the best_route.png file, and the second one is necessary to display the result with colored text in the terminal, highlighting which ecopoints were penalized.

The code will generate a .csv file named Logbook.csv that contains the algorithm's statistics (only for control usage and to more easily extract the max, avg, and min values for later plotting). It will also generate two .png files: one named max_avg_min.png, which represents the evolution of the route lengths, and the other named best_route.png, which represents the best route obtained in an anti-clockwise direction.

Inside the folder, there is another file named generate_ecopoint_list.py that was used to generate random ecopoint lists in .csv format so we can test several scenarios.