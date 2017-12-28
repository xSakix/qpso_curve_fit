from quantum_particle import QuantumParticle
import numpy as np


class QPSOCurveFit:
    def __init__(self, population_size, iterations, m=2):
        self.M = m
        self.population_size = population_size
        self.iterations = iterations
        self.population = []
        for i in range(self.population_size):
            self.population.append(QuantumParticle(self.M))

    def run(self, data, t):

        for i in range(self.iterations):
            sum_of_weights = np.zeros(self.M)
            for p in self.population:
                sum_of_weights = np.add(sum_of_weights, p.p_w)
            c = np.divide(sum_of_weights, float(self.population_size))
            for p in self.population:
                p.c = c
                p.compute_weights()
                p.compute_fitness(data, t)

            self.population.sort(key=lambda particle: particle.fitness)

            for p in self.population:
                p.g_w = self.population[0].w

            print('iteration(%d) = %f | %s' % (i,self.population[0].fitness, str(self.population[0].w)))

        return self.population[0]
