class GeneticAlgorithm:
    def __init__(self, target_img=None, init_pop=100, num_gens=1000, new_per_gen=10, mut_pct=0.1, checkpoints=2):
        """Initialize the genetic algorithm.
        
        target_img: target image of this algorithm. All individuals will
                    try to look like this target image.
        init_pop: number of individuals in initial population.
        num_gens: number of generations to simulate.
        new_per_gen: new individuals to create in each generation.
        """
        self.mut_pct = mut_pct
        self.init_pop = init_pop
        self.num_gens = num_gens
        self.new_per_gen = new_per_gen
        self.population = []
        self.target_img = target_img
        self.target_chrom = img2chromosome(target_img)
        self.target_chrom_length = len(self.target_chrom)
        self.target_chrom_length_half = self.target_chrom_length // 2
        self.best_ind = Individual(self.target_chrom)
        self.max_fitness = np.sum(self.best_ind.chromosome)
        self.checkpoints = checkpoints

    def init_population(self):
        """Create the initial individuals of the population.
        """
        for _ in range(self.init_pop):
            new_chrom = random_img(img_size=self.target_img.shape[0])
            new_chrom = img2chromosome(new_chrom)
            self.population.append(Individual(new_chrom))

    def calc_fitness(self, ind):
        """Calculate and update fitness for the individual ind.

        ind: individual to calculate fitness.
        """
        quality = np.mean(np.abs(self.target_chrom - ind.chromosome))
        quality = np.sum(self.target_chrom) - quality
        ind.fitness = quality
    
    def update_population_fitness(self):
        """Updates the fitness of every individual in the population.
        """
        for ind in self.population:
            if ind.fitness == 0:
                self.calc_fitness(ind)

    def sort_population_by_fitness(self):
        """Sorts the current population by fitness.
        """
        self.population = sorted(
            self.population,
            key=lambda i: i.fitness
        )

    def create_parent_pairs(self):
        """Selects best parents of current population.

        Returns a list of pairs of integers. Each integer representing
        the index of the parent in the current population.
        """
        pairs = []
        while len(pairs) < self.new_per_gen:
            a = np.random.randint(0, self.new_per_gen)
            b = np.random.randint(0, self.new_per_gen)
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
            self.calc_fitness(new_ind)
            new_individuals.append(new_ind)

        return new_individuals

    def crossover(self, p0, p1):
        """Create a new offspring with parents p0 and p1
        p0: parent 0
        p1: parent 1

        first = first 256 pixels of parent 0
        second = first 256 pixels of parent 1
        New individual has chromosome: [first + second]

        Returns a new individual.
        """
        p0 = self.population[p0].chromosome
        p1 = self.population[p1].chromosome
        
        # new chromosome filled with zeros
        new_chrom = np.zeros(self.target_chrom_length, dtype=np.uint8)

        rand_index = np.random.randint(0, self.target_chrom_length)
        p0_half = p0[:rand_index]
        p1_half = p1[rand_index:]
        new_chrom[:rand_index] = p0[:rand_index]
        new_chrom[rand_index:] = p1[rand_index:]
        
        # new_chrom[:self.target_chrom_length_half] = p0[:self.target_chrom_length_half]
        # new_chrom[self.target_chrom_length_half:] = p1[self.target_chrom_length_half:]

        return Individual(new_chrom)

    def mutate(self, ind):
        """Mutates the individual "ind" (in place) with certain probability.

        ind: individual to mutate.
        p: probability of mutation.
        """
        r = [np.uint32(np.random.random() * self.target_chrom_length) \
            for _ in range(np.uint32(self.target_chrom_length * self.mut_pct))]
        new_vals = np.uint8(np.random.random(len(r)) * 256)
        ind.chromosome[r] = new_vals

    def run_simulation(self):
        """Simulation loop.

        Returns.
        """
        gen_records = []

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
            # - sort population by fitness
            self.sort_population_by_fitness()
            # - drop worst individuals, configured on init
            self.population = self.population[self.new_per_gen:]
            # - save best
            if i % (self.num_gens // self.checkpoints) == 0:
                best = self.population[-1]
                gen_records.append(GenRecord(i, best.fitness, deepcopy(best.chromosome)))
            print(f'Finished generation {i}, best: {self.population[-1].fitness}')

        return gen_records

    def get_best_individual(self):
        return self.population[0]