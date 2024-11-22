import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from MLP import MLP
from Layer import Layer
from PSO import PSO

def run_mlp_pso():
    file_path = input("Enter the path of the CSV file or skip (default: concrete_data.csv): ").strip()
    if not file_path:
        file_path = 'concrete_data.csv'

    test_size = float(input("Enter the test size for splitting the data (e.g., 0.3 for 30%): "))
    num_particles = int(input("Enter the number of particles for PSO: "))
    iterations = int(input("Enter the number of iterations for PSO: "))
    informant_type = input("Specify informant type for the PSO (random/distance): ").strip().lower()
    step_size = float(input("Specify starting step size: "))

    df = pd.read_csv(file_path)
    data = np.array(df)
    X, y = data[:, :-1], data[:, -1]
    y = y.reshape(-1, 1)

    #Scaling
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # print(X_test.shape)
    X_train_input = X_train.T
    # X_val_input = X_val.T

    mlp = MLP()
    num_layers = int(input("Enter the number of hidden layers for the MLP(excluding input and output layers): "))
    input_dim = X_train.shape[1]  #number of features
    previous_dim = input_dim
    fixed_activations = input("Do you want to specify fixed activation functions? (yes/no): ").strip().lower()
    activation_functions = []

    if fixed_activations == 'yes':
        print("Available activation functions: relu, tanh, logistic")
        for i in range(num_layers):
            activation = input(f"Enter activation function for Layer {i + 1}: ").strip().lower()
            activation_functions.append(activation)

    for i in range(num_layers):
        layer_dim = int(input(f"Enter the number of neurons for Layer {i + 1}: "))
        activation = activation_functions[i] if fixed_activations == 'yes' else None
        mlp.add_layer(Layer(previous_dim, layer_dim, activation))
        previous_dim = layer_dim

    # layer1 = Layer(8, 3, None)
    # layer2 = Layer(3, 5, None)
    # layer3 = Layer(5, 8, None)
    # layer4 = Layer(8, 1, None)

    mlp.add_layer(Layer(previous_dim, 1, None)) #output layer
    # mlp.add_layer(layer1)
    # mlp.add_layer(layer2)
    # mlp.add_layer(layer3)
    # mlp.add_layer(layer4)

    total_dimensions = mlp.get_and_set_weights().size


    def mlp_cost(weights):
        mlp.get_and_set_weights(weights)
        mae = mlp.evaluate(X_train_input, y_train.T)
        return mae

    # Calculate number of layers with activations
    num_activations = len(mlp.layer_outputs) - 1

    pso = PSO(num_particles=num_particles, dimensions=total_dimensions, fitness_function=mlp_cost, num_activations=num_activations,
            informant_type=informant_type, epsilon=step_size, k=6)

    best_solutions, best_value = pso.optimize(iterations=iterations, reset_threshold=150)
    # Set final weights and print chosen activations
    mlp.get_and_set_weights(best_solutions)

    for i, layer in enumerate(mlp.layer_outputs[:-1]):
        idx = layer.activation_index
        chosen = "ReLU" if idx <= 0.33 else "Tanh" if idx <= 0.66 else "Logistic"
        print(f"Layer {i + 1} activation: {chosen} (index: {idx:.3f})")

    # activation_names = ["ReLU", "Tanh", "Logistic"]

    train_pred = mlp.forward(X_train_input)
    train_mae = np.mean(np.abs(train_pred.T - y_train))
    print(f"Training Mean Absolute Error : {train_mae:.4f}")

    y_pred = mlp.forward(X_test.T)
    # print("pred values ", y_pred[:10])
    test_mae = np.mean(np.abs(y_pred.T - y_test))

    print("\nSample Predictions vs Actuals:")
    print("Predicted  |  Actual   |  Error")
    print("-" * 35)
    for pred, true in zip(y_pred.T[:5], y_test[:5]):
        error = abs(pred[0] - true[0])
        print(f"{pred[0]:8.2f}  |  {true[0]:8.2f}  |  {error:8.2f}")

    print(f"Test Mean Absolute Error: {test_mae:.9f}")
    print(f"Best value: {best_value:.9f}")


if __name__ == "__main__":
    run_mlp_pso()