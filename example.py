from PSO import PSO

import numpy as np
import matplotlib.pyplot as plt

def sphere(x):
    return np.sum(x ** 2)

def example():
    pso = PSO(
        num_particles=20,
        dimensions=2,
        fitness_function=sphere,
        num_activations=0,
        epsilon=0.001,
        informant_type='distance',
        k=6
    )

    best_position, best_value = pso.optimize(iterations=250)

    print(f"Best position: {best_position}")
    print(f"Best value: {best_value:.9f}")


if __name__ == "__main__":
    example()