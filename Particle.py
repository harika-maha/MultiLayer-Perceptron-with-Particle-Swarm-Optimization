import numpy as np
'''
notes:
Every particle in the PSO population is initialised as an Object of this class.
A particle should have all these attributes
Initially, the attributes candidate solution(1d array?) and velocity are set to random, personal best value and current values are set to infinity, personal best position would be the same as current position.
The informant list stores the informants for each particle.
The dimensions will be the total number of parameters to be optimised by the PSO.
For the ease of understandability, the name 'position' is 'candidate_solution'
'''
# np.random.seed(42)

class Particle:
    def __init__(self, dimensions, num_activations):
        self.candidate_solution = np.random.uniform(-5, 5, dimensions)
        if num_activations > 0:
            activation_values = np.array([0.16, 0.5, 0.83])
            self.candidate_solution[-num_activations:] = np.random.choice(activation_values, size=num_activations)
        self.velocity = np.zeros(dimensions)
        if num_activations > 0:
            self.velocity[:-num_activations] = np.random.uniform(-0.5, 0.5, dimensions-num_activations)
            self.velocity[-num_activations:] = np.random.uniform(-0.1, 0.1, num_activations)    #lesser velocity for activations
        else:
            self.velocity = np.random.uniform(-0.5, 0.5, dimensions)
        self.personal_best_solution = self.candidate_solution.copy()
        self.personal_best_value = float('inf')
        self.current_value = float('inf')
        self.informants = []