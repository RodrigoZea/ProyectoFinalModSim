from individual import Individual
import numpy as np
import utils

class GeneticAlgorithm:
    def __init__(self, target_img, init_pop=100, num_gens=1000, new_per_gen=10):
        """Initialize the genetic algorithm.
        
        target_img: target image of this algorithm. All individuals will
                    try to look like this target image.
        init_pop: number of individuals in initial population.
        num_gens: number of generations to simulate.
        new_per_gen: new individuals to create in each generation.
        """
        self.init_pop = init_pop
        self.num_gens = num_gens
        self.new_per_gen = new_per_gen
        self.population = []
        self.target_chrom = utils.img2chromosome(target_img)

    def init_population(self):
        """Create the initial individuals of the population.
        """
        for _ in range(self.init_pop):
            new_chrom = utils.random_img()
            new_chrom = utils.img2chromosome(new_chrom)
            self.population.append(Individual(new_chrom))

    def calc_fitness(self, ind):
        """Calculate and update fitness for the individual ind.

        ind: individual to calculate fitness.
        """
        mean = np.mean(np.abs(self.target_chrom - ind.chromosome))
        ind.fitness = np.sum(self.target_chrom) - mean
    
    def update_population_fitness(self):
        """Updates the fitness of every individual in the population.
        """
        for ind in self.population:
            self.calc_fitness(ind)

    def sort_population_by_fitness(self):
        """Sorts the current population by fitness.
        """
        self.population = sorted(
            self.population,
            key=lambda i: i.fitness
        )

    def create_parent_pairs(self, n_parents=5):
        """Selects best parents of current population.

        n_pairs: number of pairs of parents to create.
        n_parents: number of top parents to use in pair creation.

        Returns a list of pairs of integers. Each integer representing
        the index of the parent in the current population.
        """
        pairs = []
        while len(pairs) < self.new_per_gen:
            a = np.random.randint(0, n_parents)
            b = np.random.randint(0, n_parents)
            if a == b:
                continue
            pairs.append((a, b))

        return pairs

    def create_offspring_from_pairs(self, pairs):
        """Create new individuals from the provided list of parent pairs.

        Returns a list of new individuals.
        """
        new_individuals = []
        for pair in pairs:
            new_ind = self.crossover(pair[0], pair[1])
            self.mutate(new_ind)
            new_individuals.append(new_ind)

        return new_individuals

    def crossover(self, p0, p1):
        """Create a new offspring with parents p0 and p1
        p0: parent 0
        p1: parent 1

        Returns a new individual.
        """
        new = Individual(utils.img2chromosome(utils.random_img()))

        return new

    def mutate(self, ind, p=0.1):
        """Mutates the individual "ind" (in place) with certain probability.

        ind: individual to mutate.
        p: probability of mutation.
        """
        # if np.random.random() < p:

    def run_simulation(self):
        """Simulation loop.
        """
        # initialize population
        self.init_population()
        # calculate initial fitness
        self.update_population_fitness()
        # sort initial population by fitness
        self.sort_population_by_fitness()

        # simulation loop
        for i in range(self.num_gens):
            # - select parents (best individuals)
            pairs = self.create_parent_pairs()
            # - create offspring (crossover and mutation)
            new = self.create_offspring_from_pairs(pairs)
            # - insert offspring in population
            self.population = self.population + new
            # - fitness calculation
            self.update_population_fitness()
            # - sort population by fitness
            self.sort_population_by_fitness()
            # - drop worst individuals, configured on init
            self.population = self.population[:-self.new_per_gen]