import random
from deap import base, creator, tools, algorithms

# Definiujemy minimalizację jako typ problemu
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Inicjalizacja indywidualnych rozwiązań
def create_individual():
    return [random.uniform(-10, 10)]  # Jeden parametr x w zakresie [-10, 10]

# Funkcja oceny (f(x) = x^2)
def eval_function(individual):
    x = individual[0]
    return x**2,

# Narzędzia algorytmu genetycznego
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Parametry algorytmu
population = toolbox.population(n=100)
n_generations = 40

# Uruchomienie algorytmu genetycznego
result, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_generations, verbose=False)

# Wynik
best_individual = tools.selBest(result, k=1)[0]
print(f"Najlepsze rozwiązanie: {best_individual}, wartość funkcji: {eval_function(best_individual)[0]}")
