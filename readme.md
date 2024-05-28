# Changes made to the original code:

1. Added XML parsing to read input data
2. Modified the fitness function to be able to work with multiple routes and vehicle capacities. 
3. Solution generation is now able to construct multiple routes per vehicle. 
	1. This involved initializing ants to start at the depot and generating solutions where each ant builds a route considering multiple vehicles and their capacities.
4. Updated pheromone update rules to consider multiple routes. 
I found out that increasing number of ants caused the algorithm to converge faster and run slower 
so 10 ants were optimal.
# Usage
```
python main.py --nants 10 --niters 200 -a 1.0 -b 3.0 --rho 0.1  -Q 100
```

You can also set the seed
```
python main.py --seed 42
```