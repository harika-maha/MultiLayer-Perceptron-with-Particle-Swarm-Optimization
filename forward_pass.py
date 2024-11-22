import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Layer import Layer
from MLP import MLP
from Cost import Cost

def run_example():
    df = pd.read_csv("concrete_data.csv")
    data = np.array(df)
    X, y = data[:, :-1], data[:, -1]
    y = y.reshape(-1, 1)
    # print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(X_test.shape)
    X_train_input = X_train.T

    # Create and run the MLP
    mlp = MLP()
    layer1 = Layer(8, 5, 'tanh')
    layer2 = Layer(5, 3, 'relu')
    layer3 = Layer(3, 5, 'relu')
    layer4 = Layer(5, 1)
    mlp.add_layer(layer1)
    mlp.add_layer(layer2)
    mlp.add_layer(layer3)
    mlp.add_layer(layer4)

    output = mlp.forward(X_train_input)
    mae = mlp.evaluate(X_train_input, y_train.T)
    mse = Cost(output, y_train.T).get_mse()

    print(f"Output: {output}")
    print(f"Mean Absolute Error (MAE) for train data: {mae:.4f}")
    # print(f"Mean Squared Error (MSE) for train data: {mse:.4f}")

    # print(f"Mean Absolute Error (MAE) for test data: {mae:.4f}")
    # print(f"Mean Squared Error (MSE) for test data: {mse:.4f}")

if __name__ == "__main__":
    run_example()