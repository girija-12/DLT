import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and normalize dataset
data = datasets.load_breast_cancer()
X, Y = data.data.T, data.target.reshape(1, -1)

scaler = StandardScaler()
X = scaler.fit_transform(X.T).T

# Activation functions and derivatives
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_derivative(a): return a * (1 - a)
def tanh(z): return np.tanh(z)
def tanh_derivative(a): return 1 - np.power(a, 2)

# Initialize weights and biases
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

# Forward propagation
def forward_propagation(X, parameters):
    W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return A2, {"A1": A1, "A2": A2}

# Loss function
def compute_loss(A2, Y):
    m = Y.shape[1]
    return -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m

# Backpropagation
def back_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    A1, A2 = cache["A1"], cache["A2"]
    W2 = parameters["W2"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * tanh_derivative(A1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

# Update parameters
def update_parameters(parameters, grads, lr):
    parameters["W1"] -= lr * grads["dW1"]
    parameters["b1"] -= lr * grads["db1"]
    parameters["W2"] -= lr * grads["dW2"]
    parameters["b2"] -= lr * grads["db2"]
    return parameters

# Predict
def predict(X, parameters):
    A2, _ = forward_propagation(X, parameters)
    return (A2 > 0.5).astype(int)

# Model training
def model(X, Y, hidden_units=10, iterations=2000, learning_rate=0.05, print_loss=True):
    n_x = X.shape[0]
    n_y = 1
    parameters = initialize_parameters(n_x, hidden_units, n_y)
    losses = []

    for i in range(iterations):
        A2, cache = forward_propagation(X, parameters)
        loss = compute_loss(A2, Y)
        grads = back_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            losses.append(loss)
            if print_loss:
                print(f"Loss after iteration {i}: {loss:.4f}")

    plt.plot(np.arange(0, iterations, 100), losses)
    plt.title("Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    return parameters

# Split and run model
X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size=0.2, random_state=1)
X_train, X_test = X_train.T, X_test.T
Y_train, Y_test = Y_train.T, Y_test.T

parameters = model(X_train, Y_train, hidden_units=10, iterations=2000, learning_rate=0.05)

Y_pred_train = predict(X_train, parameters)
Y_pred_test  = predict(X_test, parameters)

print(f"\nTrain accuracy: {100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100:.2f}%")
print(f"Test accuracy : {100 - np.mean(np.abs(Y_pred_test  - Y_test))  * 100:.2f}%")