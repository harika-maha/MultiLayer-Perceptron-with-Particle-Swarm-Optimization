import numpy as np

from Activations import Activations

# np.random.seed(42)


class Layer:
    def __init__(self, current_nodes, output_nodes, activation=None):
        self.current_nodes = current_nodes
        self.output_nodes = output_nodes
        self.activation = activation
        self.weights = np.random.randn(output_nodes, current_nodes) * np.sqrt(2.0/current_nodes)    #He Initialisation - sqrt(2/fan_in)*random
        self.biases = np.zeros((output_nodes,1))
        self.activation_index = 0 if not activation else self.get_activation_index(activation)

    def get_activation_index(self, activation):
        activation_map = {
            'relu': 0.16,  # Middle of ReLU range
            'tanh': 0.5,  # Middle of tanh range
            'logistic': 0.83  # Middle of logistic range
        }
        return activation_map.get(activation.lower(), 0)

    def get_activation(self, Z):
        #Printing current activation
        if self.output_nodes == 1:
            print("Activation: Output")
        elif self.activation:
            print(f"Activation: Fixed {self.activation}")
        else:
            chosen = "RELU" if self.activation_index <= 0.33 else "TANH" if self.activation_index <= 0.66 else "LOGISTIC"
            print(f"Activation: PSO chose {chosen} (index: {self.activation_index:.3f})")
        #For output layer
        if self.output_nodes == 1:
            return Z
        elif self.activation:   #for fixed activations
            activation_funcs = {
                'relu': Activations().relu,
                'tanh': Activations().tanh,
                'logistic': Activations().logistic
            }
            return activation_funcs[self.activation.lower()](Z)
        else:   #determined by PSO
            # equally split index
            if self.activation_index <= 0.33:
                return Activations().relu(Z)
            elif self.activation_index <= 0.66:
                return Activations().tanh(Z)
            else:
                return Activations().logistic(Z)

    def forward_pass(self, X):
        self.Z = np.dot(self.weights, X) + self.biases
        return self.get_activation(self.Z)