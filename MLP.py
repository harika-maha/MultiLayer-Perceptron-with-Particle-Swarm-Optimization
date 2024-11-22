import numpy as np
from Cost import Cost

class MLP:
    def __init__(self):
        self.layer_outputs = []

    def add_layer(self, layer):
        self.layer_outputs.append(layer)

    def get_and_set_weights(self, weights=None):
        if weights is None:
            flat_weights = []
            activation_positions = []
            current_position = 0
            for layer in self.layer_outputs:
                weight_size = layer.weights.size
                flat_weights += list(layer.weights.flatten())
                current_position += weight_size
                bias_size = layer.biases.size
                flat_weights += list(layer.biases.flatten())
                current_position += bias_size
                if layer is not self.layer_outputs[-1]:  # Skip activation for output layer
                    flat_weights.append(layer.activation_index)
                    activation_positions.append(current_position)   #tracking the positions where activation indices are stored in array
                    current_position += 1
            return np.array(flat_weights)
        else:
            index = 0
            for layer in self.layer_outputs:
                weight_size = layer.weights.size
                bias_size = layer.biases.size
                layer.weights = weights[index:index + weight_size].reshape(layer.weights.shape) #select the layer's weights
                index += weight_size    #move to the biases
                layer.biases = weights[index:index + bias_size].reshape(layer.biases.shape) #select the layer's biases
                index += bias_size  #move to the next weights
                if layer is not self.layer_outputs[-1]:
                    layer.activation_index = weights[index]     #the third part is the activation index
                    # Ensure activation index stays in [0,1] range
                    layer.activation_index = np.clip(layer.activation_index, 0, 1)
                    index += 1  #move to the next layer's weights

    def forward(self, X):
        for layer in self.layer_outputs:
            # print(f'X shape before forward_prop {X.shape}')
            X = layer.forward_pass(X)
            # print("X shape after forward_prop", X.shape)
        return X

    def evaluate(self, X, y):
        output = self.forward(X)
        cost_instance = Cost(output, y)
        return cost_instance.get_mae()

    def objective_function(self, weights, X, y):
        self.get_and_set_weights(weights)
        return self.evaluate(X, y)