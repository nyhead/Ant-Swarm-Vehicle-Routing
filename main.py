import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nants', type=int,default=10,required=False)
parser.add_argument('--niters', type=int,default=200,required=False)
parser.add_argument('-a', type=float,default=1.0,required=False)
parser.add_argument('-b', type=float,default=3.0,required=False)
parser.add_argument('--rho', type=float,default=0.1,required=False)
parser.add_argument('-Q', type=int,default=100,required=False)
parser.add_argument('--seed', type=int,required=False)
args = parser.parse_args()

num_ants = args.nants
num_iterations = args.niters
alpha = args.a  # Pheromone importance
beta = args.b   # Heuristic importance
rho = args.rho   # Pheromone evaporation rate
Q = args.Q   # Pheromone deposit factor

if args.seed:
    random.seed(args.seed)

def parse_vrp_xml(file_path):
    tree = ET.parse(file_path)

    root = tree.getroot()

    # Extract node information
    nodes = []
    for node in root.find('network').find('nodes'):
        node_id = int(node.attrib['id'])
        node_type = int(node.attrib['type'])
        cx = float(node.find('cx').text)
        cy = float(node.find('cy').text)
        nodes.append((node_id, node_type, cx, cy))

    # Extract vehicle information
    vehicle = root.find('fleet').find('vehicle_profile')
    vehicle_capacity = float(vehicle.find('capacity').text)
    depot_id = int(vehicle.find('departure_node').text)
    # print(depot_id)
    # Extract request information
    requests = []
    for request in root.find('requests'):
        request_id = int(request.attrib['id'])
        node_id = int(request.attrib['node'])
        quantity = float(request.find('quantity').text)
        requests.append((request_id, node_id, quantity))

    return nodes, vehicle_capacity, requests




def initialize_ants(num_ants, depot_index, num_customers):
    return [[depot_index] for _ in range(num_ants)]

def calculate_probabilities(current_node, unvisited, pheromone_matrix, dist_matrix, alpha, beta):
    pheromone = np.array([pheromone_matrix[current_node][j] for j in unvisited])
    heuristic = np.array([1.0 / dist_matrix[current_node][j] for j in unvisited])
    pheromone = pheromone ** alpha
    heuristic = heuristic ** beta
    probabilities = pheromone * heuristic
    probabilities /= probabilities.sum()
    return probabilities

def construct_solution(pheromone_matrix, dist_matrix, num_ants, alpha, beta, depot_index, num_customers):
    solutions = initialize_ants(num_ants, depot_index, num_customers)
    for ant in solutions:
        unvisited = list(range(1, num_customers + 1))
        while unvisited:
            current_node = ant[-1]
            probabilities = calculate_probabilities(current_node, unvisited, pheromone_matrix, dist_matrix, alpha, beta)
            next_node = random.choices(unvisited, probabilities)[0]
            ant.append(next_node)
            unvisited.remove(next_node)
        ant.append(depot_index)  # Return to depot
    return solutions

def calculate_route_length(route, dist_matrix):
    length = 0
    for i in range(len(route) - 1):
        length += dist_matrix[route[i]][route[i + 1]]
    return length

def update_pheromones(pheromone_matrix, solutions, dist_matrix, rho, Q):
    pheromone_matrix *= (1 - rho)
    for solution in solutions:
        route_length = calculate_route_length(solution, dist_matrix)
        for i in range(len(solution) - 1):
            pheromone_matrix[solution[i]][solution[i + 1]] += Q / route_length

def aco_vrp(dist_matrix, num_ants, num_iterations, alpha, beta, rho, Q, depot_index, num_customers):
    pheromone_matrix = np.ones((len(dist_matrix), len(dist_matrix)))
    best_solution = None
    best_length = float('inf')
    length_history = [] 
    
    for iteration in range(num_iterations):
        solutions = construct_solution(pheromone_matrix, dist_matrix, num_ants, alpha, beta, depot_index, num_customers)
        lengths = [calculate_route_length(solution, dist_matrix) for solution in solutions]
        
        min_length = min(lengths)
        if min_length < best_length:
            best_length = min_length
            best_solution = solutions[lengths.index(min_length)]
            
        length_history.append(best_length)
        update_pheromones(pheromone_matrix, solutions, dist_matrix, rho, Q)
    
    return best_solution, best_length, length_history




def run(filename):
    print()
    print(f'{filename}:')
    nodes, vehicles, demands = parse_vrp_xml(filename)

    # print(nodes, vehicles, demands)

    coordinates = np.array([(x, y) for _, _, x, y in nodes])
    demands = np.array(demands)

    # Distance matrix
    num_nodes = len(coordinates)
    dist_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(coordinates[i] - coordinates[j])

    # Initialize pheromone levels
    pheromone_matrix = np.ones((num_nodes, num_nodes))

    depot_index = 0
    num_customers = len(nodes) - 1

    best_solution, best_length,length_history = aco_vrp(dist_matrix, num_ants, num_iterations, alpha, beta, rho, Q, depot_index, num_customers)
    print("Distance history:")
    print(length_history)
    print("Best solution and best length:")
    print(best_solution, best_length)
    return best_solution, best_length,length_history

if __name__ == '__main__':
    bs32, bl32, lh32 = run("hw3/data_32.xml")
    bs72, bl72, lh72 = run("hw3/data_72.xml")
    bs422, bl422, lh422 = run("hw3/data_422.xml")
    
    results = [(bs32, bl32, lh32), (bs72, bl72, lh72), (bs422, bl422, lh422)]    
    plt.figure(figsize=(10, 6))
    for idx, (best_solution, best_length, length_history) in enumerate(results):
        plt.plot(length_history, label=f'Instance {idx+1}')

    plt.title('Convergence of ACO Algorithm for Different VRP Instances')
    plt.xlabel('Iteration')
    plt.ylabel('Best Solution Length')
    plt.legend()
    plt.grid(True)
    plt.show()