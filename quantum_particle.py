import numpy as np
from sklearn.metrics import mean_squared_error


class QuantumParticle:
    def __init__(self, m=2):
        self.M = m
        self.w = np.random.uniform(-5., 5., m)
        self.p_w = np.array(self.w)
        self.g_w = np.array(self.w)
        self.c = np.array(self.w)
        self.alpha = 0.75
        self.last_fitness = None
        self.fitness = None

    def compute_weights(self):
        phi = np.random.uniform(0., 1.)
        p = np.add(np.multiply(phi, self.p_w), np.multiply(np.subtract(1., phi), self.g_w))
        u = np.random.uniform(0., 1.)
        for i in range(len(self.w)):
            if np.random.uniform(0., 1.) < 0.5:
                self.w[i] = p[i] + self.alpha * np.abs(self.w[i] - self.c[i]) * np.log(1. / u)
            else:
                self.w[i] = p[i] - self.alpha * np.abs(self.w[i] - self.c[i]) * np.log(1. / u)

    def evaluate(self, x):
        return np.polyval(self.w, x)

    def compute_fitness(self, data, t):
        y = self.evaluate(data)
        self.fitness = mean_squared_error(t, y)

        if self.last_fitness is None or self.last_fitness > self.fitness:
            self.last_fitness = self.fitness
            self.p_w = self.w
