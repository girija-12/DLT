import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
data = datasets.load_breast_cancer()
X, Y = data.data, data.target.reshape(1, -1)

# 2. Normalize (standardize) features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X, Y = X_std.T, Y 

# 3. Initialize parameters
def initialize(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b

# 4. Sigmoid and propagation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = np.mean(A - Y)
    return {"dw": dw, "db": db}, cost

# 5–6. Optimize with gradient descent
def optimize(w, b, X, Y, num_iter=2000, lr=0.01, print_cost=False):
    costs = []
    for i in range(num_iter):
        grads, cost = propagate(w, b, X, Y)
        w -= lr * grads["dw"]
        b -= lr * grads["db"]
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Iteration {i:4d}: cost = {cost:.4f}")
    return (w, b), costs

# 7. Predict
def predict(w, b, X):
    A = sigmoid(np.dot(w.T, X) + b)
    return (A > 0.5).astype(int)

def model(X, Y, test_size=0.2, num_iter=2000, lr=0.01, print_cost=True):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X.T, Y.T, test_size=test_size, random_state=1)
    X_train, X_test = X_train.T, X_test.T
    Y_train, Y_test = Y_train.T, Y_test.T

    w, b = initialize(X_train.shape[0])
    (w, b), costs = optimize(w, b, X_train, Y_train,
                             num_iter, lr, print_cost)

    Y_pred_train = predict(w, b, X_train)
    Y_pred_test  = predict(w, b, X_test)

    train_acc = 100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100
    test_acc  = 100 - np.mean(np.abs(Y_pred_test  - Y_test)) * 100

    print(f"\nTrain accuracy: {train_acc:.2f}%")
    print(f"Test accuracy : {test_acc:.2f}%")

    return {
        "w": w, "b": b, "costs": costs,
        "Y_pred_test": Y_pred_test, "Y_pred_train": Y_pred_train
    }

model_info = model(X, Y, num_iter=2000, lr=0.05, print_cost=True)