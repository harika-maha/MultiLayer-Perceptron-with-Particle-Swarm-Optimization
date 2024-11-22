import numpy as np
from Particle import Particle
import matplotlib.pyplot as plt
from pso_parameters import pso_params, reset_parameters, update_parameters

class PSO:
    def __init__(self, num_particles, dimensions, fitness_function, num_activations, epsilon=pso_params['epsilon'], informant_type='random', k=6):
        self.particles = []
        self.num_activations = num_activations
        for n in range(num_particles):
            self.particles.append(Particle(dimensions, num_activations))                 #stores all the particles of the population
        self.dimensions = dimensions
        self.fitness_function = fitness_function                    #fitness function to evaluate
        self.global_best_solution = None
        self.global_best_value = float('inf')
        self.history = []                                           #history of global best values
        self.epsilon = epsilon
        #informant selection
        self.k = k
        self.informant_type = informant_type.lower()

    def calculate_distance(self, p1, p2):
        return np.linalg.norm(p1.candidate_solution - p2.candidate_solution)
    def assign_random_informant(self):
        for i, particle in enumerate(self.particles):
            available_indices = list(range(len(self.particles)))  # array of available indices - to choose informants from
            available_indices.remove(i)  # exclude current particle
            informant_indices = np.random.choice(available_indices, size=min(self.k, len(self.particles) - 1), replace=False)
            particle.informants = [self.particles[idx] for idx in informant_indices]

    def assign_distance_informant(self):    #knn approach
        for i, particle in enumerate(self.particles):
            distances = []
            for j, p2 in enumerate(self.particles):
                if i!=j:
                    dist = self.calculate_distance(particle, p2)
                    distances.append((dist, p2))
            distances.sort(key=lambda x: x[0])
            particle.informants = [p for id, p in distances[:self.k]]

    def informants_update(self):
        if self.informant_type == 'distance':
            self.assign_distance_informant()
        else:
            self.assign_random_informant()


    def optimize(self, iterations, reset_threshold=1000):
        iterations_without_improvement = 0
        current_best_value = float('inf')
        plt.ion()  # Enable interactive mode
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        v_max_weights = pso_params['v_max_factor'] * (10 - (-10))  # 0.1 * 20 = 2
        v_max_activations = pso_params['v_max_factor'] * (1 - 0)  # 0.1 * 1 = 0.1

        alpha = pso_params['alpha']  # inertia weight
        beta = pso_params['beta']   # cognitive weight
        gamma = pso_params['gamma']  # social weight (informants, in this case)
        delta = pso_params['delta']  # global weight
        self.epsilon = pso_params['epsilon']

        initial_epsilon = self.epsilon
        initial_alpha = 0.9
        final_alpha = 0.4

        for k in range(iterations):
            progress = k / iterations
            self.epsilon = initial_epsilon * (1 - progress * 0.9)  # step size decreases
            current_alpha = initial_alpha - progress * (initial_alpha - final_alpha)  # inertia decay from 0.9 to 0.4
            if k%update_parameters['update_frequency'] ==0:
                self.informants_update()
            previous_best = self.global_best_value
            for particle in self.particles:                     #for all dimensions/particles
                particle.current_value = self.fitness_function(particle.candidate_solution)       #current loss/fitness value (for mlp optimisation - particle position is the array of weights+biases)
                #personal best update
                if particle.current_value < particle.personal_best_value:                         #minimization
                    particle.personal_best_value = particle.current_value               #set best value to current value
                    particle.personal_best_solution = particle.candidate_solution.copy()          #set best position to current values
                #global best update
                if particle.current_value < self.global_best_value:                     #if current value is better than global best
                    self.global_best_value = particle.current_value                     #update value and position
                    self.global_best_solution = particle.candidate_solution.copy()

                best_informant = min(particle.informants, key=lambda p: p.personal_best_value)   #best among the informants
                r1, r2, r3 = np.random.random(3)
                # new velocity = inertia_component + cognitive_component + social_component + global_component
                new_velocity = (current_alpha * particle.velocity + beta * r1 * (particle.personal_best_solution - particle.candidate_solution) + gamma * r2 * (best_informant.personal_best_solution - particle.candidate_solution) + delta * r3 * (self.global_best_solution - particle.candidate_solution))
                particle.velocity = new_velocity        #velocity update rule
                if self.num_activations > 0:
                    particle.velocity[:-self.num_activations] = np.clip(particle.velocity[:-self.num_activations], -v_max_weights, v_max_weights)
                    particle.velocity[-self.num_activations:] = np.clip(particle.velocity[-self.num_activations:], -v_max_activations, v_max_activations)
                else:
                    particle.velocity = np.clip(particle.velocity, -v_max_weights, v_max_weights)   #for layer without activation
                particle.candidate_solution += self.epsilon * particle.velocity  #update using step_size

                if self.num_activations > 0:
                    particle.candidate_solution[:-self.num_activations] = np.clip(particle.candidate_solution[:-self.num_activations], -5, 5)   #clipping weights normally
                    # particle.candidate_solution[-self.num_activations:] = np.clip(particle.candidate_solution[-self.num_activations:], -5, 5)
                    # particle.candidate_solution[-self.num_activations] = np.clip(particle.candidate_solution[-self.num_activations], 0, 0.66)
                    activation_indices = particle.candidate_solution[-self.num_activations:]
                    for i in range(len(activation_indices)):
                        activation_indices[i] = np.clip(activation_indices[i], 0, 1) #clipping activation function from 0 to 1
                        if activation_indices[i] < 0.33:
                            activation_indices[i] = 0.16
                        elif activation_indices[i] < 0.66:
                            activation_indices[i] = 0.5
                        else:
                            activation_indices[i] = 0.83

            if self.global_best_value < previous_best - reset_parameters['improvement_threshold']:
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
            if iterations_without_improvement >= reset_threshold:
                print(f'Resetting at iteration {k} with best value {self.global_best_value}')
                best_solution = self.global_best_solution.copy()

                for p in self.particles:
                    spread = np.random.uniform(reset_parameters['min_spread'], reset_parameters['max_spread'])  # Variable spread for different particles
                    p.candidate_solution = best_solution + np.random.uniform(-spread, spread, self.dimensions)
                    p.velocity = np.random.uniform(*reset_parameters['velocity_range'], self.dimensions)
                iterations_without_improvement = 0

            self.history.append(self.global_best_value)

            #visualisation of the particles
            ax1.clear()
            x = [p.candidate_solution[0] for p in self.particles]
            y = [p.candidate_solution[1] for p in self.particles]
            ax1.scatter(x, y, c='blue', label='Particles')
            if self.global_best_solution is not None:
                ax1.scatter(self.global_best_solution[0], self.global_best_solution[1], c='red', marker='*', s=200,
                                label='Global Best')
            ax1.set_title(f'Particle Positions (Iteration {k})')
            ax1.set_xlim(-10, 10)
            ax1.set_ylim(-10, 10)
            ax1.legend()
            ax1.grid(True)
            ax2.clear()
            ax2.plot(self.history)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Best Value')
            ax2.set_title('PSO Convergence')
            ax2.grid(True)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)  # Pause

        plt.ioff()
        plt.show()

        return self.global_best_solution, self.global_best_value