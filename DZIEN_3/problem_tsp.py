import random
import numpy as np
from deap import base, creator, tools

# Odległości między miastami (symulacja)
distances = np.random.rand(10, 10)  # Macierz odległości dla 10 miast
for i in range(10):
    distances[i, i] = 0  # Odległość do samego siebie to 0

# Tworzenie funkcji fitness i osobnika
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimalizujemy długość trasy
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Generator osobników - permutacja miast
toolbox.register("indices", random.sample, range(10), 10)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)

# Generator populacji
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Funkcja przystosowania - suma odległości między kolejnymi miastami w trasie
def evalTSP(individual):
    total_distance = 0
    for i in range(len(individual) - 1):
        total_distance += distances[individual[i], individual[i+1]]
    total_distance += distances[individual[-1], individual[0]]  # Powrót do miasta startowego
    return total_distance,

toolbox.register("evaluate", evalTSP)

# Operator krzyżowania
toolbox.register("mate", tools.cxPartialyMatched)

# Operator mutacji - zamiana dwóch miast
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

# Operator selekcji
toolbox.register("select", tools.selTournament, tournsize=3)

# Funkcja główna
def main():
    random.seed(42)
    pop = toolbox.population(n=300)  # Tworzymy populację 300 osobników
    CXPB, MUTPB = 0.7, 0.2  # Prawdopodobieństwo krzyżowania i mutacji

    # Ewaluacja początkowej populacji
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Algorytm genetyczny: 100 pokoleń
    for gen in range(100):
        # Selekcja najlepszych osobników
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Krzyżowanie
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutacja
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Ewaluacja nowych osobników
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Zastąpienie starej populacji nową
        pop[:] = offspring

    # Znalezienie najlepszego rozwiązania
    best_ind = tools.selBest(pop, 1)[0]
    print(f"Najlepsza trasa: {best_ind}")
    print(f"Długość najlepszej trasy: {best_ind.fitness.values[0]}")

if __name__ == "__main__":
    main()
