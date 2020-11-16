from individual import Individual

class GeneticAlgorithm:
    def __init__(self, target_img, init_pop=100, n_iters=1000):
        """Initialize the genetic algorithm.
        
        target_img: target image of this algorithm. All individuals will
                    try to look like this target image.
        init_pop: number of individuals in initial population.
        n_iters: number of iterations in simulation loop
        """
        self.init_pop = init_pop
        self.n_iters = n_iters
        self.target_img = target_img
        self.population = []

    def init_population():
        """Create the initial individuals of the population.
        """
        for i in range(self.init_pop):
            pass
        pass

    def select_parents(n_pairs=10):
        """Selects best parents of current population.

        n_pairs: number of pairs of parents to create.

        Returns a list of pairs of parents.
        """
        pass

    def crossover(p0, p1):
        """Create a new offspring with parents p0 and p1
        p0: parent 0
        p1: parent 1

        Returns a new individual.
        """
        pass

    def mutate(ind, p):
        """Mutates the individual "ind" (in place) with certain probability.

        ind: individual to mutate.
        p: probability of mutation.
        """
        pass

    def run_simulation():
        """Simulation loop.
        """
        # initialize population
        # calculate initial fitness

        # simulation loop
        for i in range(self.n_iters):
            # - select parents (best individuals)
            # - create offspring (crossover and mutation)
            # - insert offspring in population
            # - fitness calculation
            # - drop worst individuals
            pass
        pass