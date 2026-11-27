import numpy as np


#Distances between the depot and delivery locations (in kilometres).
locations = ["A", "B", "C", "D", "E", "F", "G"]

distance_matrix = np.array([
    [0, 4, 8, 6, 7, 9, 5],
    [4, 0, 5, 7, 3, 8, 6],
    [8, 5, 0, 4, 6, 3, 7],
    [6, 7, 4, 0, 5, 6, 8],
    [7, 3, 6, 5, 0, 4, 6],
    [9, 8, 3, 6, 4, 0, 5],
    [5, 6, 7, 8, 6, 5, 0]
])

location_index = {location: index for index, location in enumerate(locations)}


def calculate_distance(route):
    complete_route = ["A"] + list(route) + ["A"]
    total_distance = 0

    for i in range(len(complete_route) - 1):
        start = location_index[complete_route[i]]
        end = location_index[complete_route[i + 1]]
        total_distance += distance_matrix[start, end]

    return total_distance


def select_parent(population, fitness, rng):
    #Select three routes and choose the one with the shortest distance.
    contestants = rng.integers(0, len(population), size=3)
    winner = contestants[np.argmin(fitness[contestants])]
    return population[winner]


def crossover(parent1, parent2, rng):
    #Order crossover keeps every delivery location exactly once.
    start, end = sorted(rng.choice(len(parent1), size=2, replace=False))
    child = [None] * len(parent1)

    child[start:end + 1] = parent1[start:end + 1]

    parent2_order = parent2[end + 1:] + parent2[:end + 1]
    remaining_locations = [
        location for location in parent2_order if location not in child
    ]

    empty_positions = (
        list(range(end + 1, len(parent1))) + list(range(start))
    )

    for position, location in zip(empty_positions, remaining_locations):
        child[position] = location

    return child


def mutate(route, rng):
    #Swap two locations to produce a different valid route.
    new_route = route.copy()
    first, second = rng.choice(len(new_route), size=2, replace=False)
    new_route[first], new_route[second] = new_route[second], new_route[first]
    return new_route


def run_genetic_algorithm(seed=42, population_size=100, generations=100):
    rng = np.random.default_rng(seed)
    delivery_locations = locations[1:]

    population = [
        list(rng.permutation(delivery_locations))
        for _ in range(population_size)
    ]

    best_route = None
    best_distance = float("inf")

    for generation in range(generations):
        fitness = np.array([calculate_distance(route) for route in population])

        best_index = np.argmin(fitness)
        if fitness[best_index] < best_distance:
            best_distance = int(fitness[best_index])
            best_route = population[best_index].copy()

        # Keep the two best routes unchanged in the next generation.
        elite_indices = np.argsort(fitness)[:2]
        new_population = [
            population[index].copy() for index in elite_indices
        ]

        while len(new_population) < population_size:
            parent1 = select_parent(population, fitness, rng)
            parent2 = select_parent(population, fitness, rng)

            if rng.random() < 0.90:
                child = crossover(parent1, parent2, rng)
            else:
                child = parent1.copy()

            if rng.random() < 0.15:
                child = mutate(child, rng)

            new_population.append(child)

        population = new_population

    #Check the last generation before returning the best route.
    final_fitness = np.array([calculate_distance(route) for route in population])
    final_index = np.argmin(final_fitness)

    if final_fitness[final_index] < best_distance:
        best_distance = int(final_fitness[final_index])
        best_route = population[final_index].copy()

    return best_route, best_distance


best_route, best_distance = run_genetic_algorithm(seed=42)

complete_route = ["A"] + best_route + ["A"]

print("Best route:", " -> ".join(complete_route))
print("Total distance:", best_distance, "km")
