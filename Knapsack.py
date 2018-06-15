'''
Andrew Zhu and Darsh Lin
Comparison of different knapsack algorithms
with google dynamic programming knapsacksolve. genetic algorithm, and our own modified genetic algorithm
'''


from __future__ import print_function
from pyeasyga import modifiedga
from pyeasyga import firstga
from ortools.algorithms import pywrapknapsack_solver
import time
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt


def optimized_dynamic(val, w, size, max_weight):
    solver = pywrapknapsack_solver.KnapsackSolver \
        (pywrapknapsack_solver.KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
         'test')

    cap = max_weight

    capacities = [cap]
    values = []
    weights = [[]]
    for i in range(size):
        values.append(val[i])
        weights[0].append(w[i])

    start = time.time()
    solver.Init(values, weights, capacities)
    computed_value = solver.Solve()
    packed_items = [x for x in range(0, len(weights[0]))
                    if solver.BestSolutionContains(x)]
    packed_weights = [weights[0][i] for i in packed_items]
    total_weight = sum(packed_weights)
    end = time.time()
    elapsed = end-start

    print("Optimized Dynamic: ")
    print("Packed items: ", packed_items)
    print("Packed weights: ", packed_weights)
    print("Total weight: ", total_weight)
    print("Total value: ", computed_value)
    print("Optimized Dynamic Time: ", elapsed)
    print("\n")
    return total_weight, computed_value, elapsed


def original_genetic(val, w, size, max_weight):
    data = []
    for i in range(size):
        data.append({'value': val[i], 'weight': w[i]})

    ga = firstga.GeneticAlgorithm(data)

    def fitness(individual, data):
        values, weights = 0, 0
        for selected, box in zip(individual, data):
            if selected:
                values += box.get('value')
                weights += box.get('weight')
        if weights > max_weight:
            values = 0
        return values

    ga.fitness_function = fitness
    start = time.time()
    ga.run()
    end = time.time()
    best_weight = 0
    packed_items = []
    packed_weights = []
    for x in range(len(ga.get_best_genes())):
        data_num = data[x].get('weight')
        data_weight = data_num
        if ga.get_best_genes()[x] > 0:
            packed_weights.append(data_weight)
            packed_items.append(x)
            best_weight += data_weight

    best_individual = ga.best_individual()

    elapsed = end-start

    print("Original Genetic: ")
    print("Packed items: ", packed_items)
    print("Packed weights: ", packed_weights)
    print("Total weight: ", best_weight)
    print("Total value: ", best_individual[0])
    print("Original Genetic Time: ", elapsed)
    print("\n")
    return best_weight, best_individual[0], elapsed


def modified_genetic(val, w, size, max_weight):
    data = []

    for i in range(size):
        data.append({'value': val[i], 'weight': w[i]})

    ga = modifiedga.GeneticAlgorithm(data)

    def fitness(individual, data):
        values, weights = 0, 0
        for selected, box in zip(individual, data):
            if selected:
                values += box.get('value')
                weights += box.get('weight')
        if weights > max_weight:
            values = 0
        return values

    ga.fitness_function = fitness
    start = time.time()
    ga.run()
    end = time.time()
    best_weight = 0
    packed_items = []
    packed_weights = []
    for x in range(len(ga.get_best_genes())):
        data_num = data[x].get('weight')
        data_weight = data_num
        if ga.get_best_genes()[x] > 0:
            packed_weights.append(data_weight)
            packed_items.append(x)
            best_weight += data_weight
    best_individual = ga.best_individual()

    elapsed = end-start

    print("Modified Genetic: ")
    print("Packed items: ", packed_items)
    print("Packed weights: ", packed_weights)
    print("Total weight: ", best_weight)
    print("Total value: ", best_individual[0])
    print("Modified Genetic Time: ", elapsed)
    print("\n")
    return best_weight, best_individual[0], elapsed


def brute_force(values, weights, size, max_weight):
    cap = max_weight
    A = [0 for x in range(size)]

    best_value = 0
    start = time.time()
    for i in range(int(math.pow(2, size))):
        j = size - 1
        temp_weight = 0
        temp_value = 0
        while A[j] != 0 and j >= 0:
            A[j] = 0
            j = j - 1
        A[j] = 1
        for k in range(size):
            if A[k] == 1:
                temp_weight = temp_weight + weights[k]
                temp_value = temp_value + values[k]
        if temp_value > best_value and temp_weight <= cap:
            best_value = temp_value
            best_choice = copy.deepcopy(A)
    end = time.time()

    items = []
    packed_weights = []
    total_weight = 0
    total_value = 0

    for f in range(len(best_choice)):
        if best_choice[f] > 0:
            items.append(f)
            packed_weights.append(weights[f])
            total_weight += weights[f]
            total_value += values[f]

    elapsed = end-start

    print("Brute Force: ")
    print("Packed Items: ", items)
    print("Packed Weights: ", packed_weights)
    print("Total weight: ", total_weight)
    print("Total value: ", total_value)
    print("Brute Force Time: ", elapsed)

    return total_weight, total_value, elapsed


def main():
    # all data created here
    size = 100
    max_weight = 250
    random.seed()
    values = []
    weights = []
    max_unrestricted_weight = 0
    max_unrestricted_value = 0

    for i in range(size):
        val = random.randint(1,20)
        values.append(val)
        max_unrestricted_value += val

    for j in range(size):
        w = random.randint(1,10)
        weights.append(w)
        max_unrestricted_weight += w


    print("Data in: ")
    print("weights: ", weights)
    print("values: ", values)
    print("Max Unrestricted Weight: ", max_unrestricted_weight)
    print("Max Unrestricted value: ", max_unrestricted_value)
    print("\n")

    od = optimized_dynamic(values, weights, size, max_weight)
    og = original_genetic(values, weights, size, max_weight)
    mg = modified_genetic(values, weights, size, max_weight)
    # 0 total weight 1 total value 2 time

    # this is to show values
    n = 3
    solution_values= (od[1], og[1], mg[1])
    muv = (max_unrestricted_value-od[1],
           max_unrestricted_value-og[1],
           max_unrestricted_value-mg[1],
           )
    ind = np.arange(n)
    width = 0.5

    p1 = plt.bar(ind, solution_values, width)
    p2 = plt.bar(ind, muv, width, bottom=solution_values)

    plt.ylabel('Values')
    plt.title('Knapsack Solution Values by Algorithm')
    plt.xticks(ind,('Optimized Dynamic', 'Original Genetic', 'Modified Genetic', ))#'Brute Force'))
    plt.yticks(np.arange(0,max_unrestricted_value + 10,max_unrestricted_value * .2))
    plt.legend((p1[0], p2[0]), ('Solution', 'Max Unrestricted'))

    plt.show()

    # this is to show time
    n = 3

    solution_values = (od[2], og[2], mg[2])  # bf[1])
    ind = np.arange(n)
    width = 0.5

    p1 = plt.bar(ind, solution_values, width)

    plt.ylabel('Times')
    plt.title('Knapsack Solution Times by Algorithm')
    plt.xticks(ind, ('Optimized Dynamic', 'Original Genetic', 'Modified Genetic',))  # 'Brute Force'))
    plt.yticks(np.arange(0, 15, .5))

    plt.show()


if __name__ == "__main__":
    main()
