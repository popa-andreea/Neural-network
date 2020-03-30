import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import math

#exercitiul 1 si 2

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([-1, 1, 1, 1])

epochs = 70
learning_rate = 0.1

def compute_y(x, W, bias):
    return (-x * W[0] - bias) / (W[1] + 1e-10)

def plot_decision_boundary(X, y , W, b, current_x, current_y):
    x1 = -0.5
    y1 = compute_y(x1, W, b)
    x2 = 0.5
    y2 = compute_y(x2, W, b)
    plt.clf()
    color = 'r'
    if(current_y == -1):
        color = 'b'
    plt.ylim((-1, 2))
    plt.xlim((-1, 2))
    plt.plot(X[y == -1, 0], X[y == -1, 1], 'b+')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')
    plt.plot(current_x[0], current_x[1], color+'s')
    plt.plot([x1, x2] ,[y1, y2], 'black')
    plt.show(block=False)
    plt.pause(0.3)

def train_perceptron(X, y, epochs, learning_rate):
    W = np.zeros(2)
    bias = 0
    no_samps = X.shape[0]
    accuracy = 0.0

    for epoch in range(epochs):
        X, y = shuffle(X, y)
        for i in range(no_samps):
            y_hat = np.dot(X[i][:],W) + bias
            loss = (y_hat - y[i]) ** 2
            
            W = W - learning_rate * (y_hat - y[i]) * X[i][:]
            bias = bias - learning_rate * (y_hat - y[i])
            
            accuracy = np.mean(np.sign(np.dot(X, W) + bias) == y)
            print("sample loss:", loss, "accuracy:", accuracy)
            
            plot_decision_boundary(X, y, W, bias, X[i][:], y[i])
    return W, bias, accuracy

W, bias, accuracy = train_perceptron(X, y, epochs, learning_rate)
print("\nExercitiul 2\n weights: ", W, "\n bias:", bias, "\n accuracy:", accuracy)

#exercitiul 3
y_ = [-1, 1, 1, -1]

W_, bias_, accuracy_ = train_perceptron(X, y_, epochs, learning_rate)
print("\nExercitiul 3\n weights: ", W_, "\n bias:", bias_, "\n accuracy:", accuracy_)

#exercitiul 4
def compute_y(x, W, bias):
    return (-x*W[0] - bias) / (W[1] + 1e-10)

def plot_decision(X_, W_1, W_2, b_1, b_2):
    plt.clf()
    plt.ylim((-0.5, 1.5))
    plt.xlim((-0.5, 1.5))
    xx = np.random.normal(0, 1, (100000))
    yy = np.random.normal(0, 1, (100000))
    X = np.array([xx, yy]).transpose()
    X = np.concatenate((X, X_))
    _, _, _, output = forward(X, W_1, b_1, W_2, b_2)
    y = np.squeeze(np.round(output))
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'b+')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')
    plt.show(block=False)
    plt.pause(0.1)

def tanh(x):
    return np.divide(np.exp(2 * x) - 1, np.exp(2 * x) + 1)

def sigmoid(x):
    return np.divide(1, 1 + np.exp(-x))

def forward(X, W1, W2, b1, b2):
    z1 = np.dot(X, W1) + b1
    a1 = tanh(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    return z1, a1, z2, a2

def tanh_derivative(x):
    return 1 - tanh(x)**2

def backward(a1, a2, z1, W2, X, y, no_samps):
    dz2 = a2 - y
    dw2 = np.dot(a1.T, dz2) / no_samps
    db2 = np.sum(dz2) / no_samps
    da1 = np.dot(dz2, W2.T)
    dz1 = np.dot(da1, tanh_derivative(z1))
    dw1 = np.dot(X.T, dz1) / no_samps
    db1 = np.sum(dz1) / no_samps

    return dw1, db1, dw2, db2

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.expand_dims(np.array([0, 1, 1, 0]), 1)

epochs = 70
learning_rate = 0.5
no_hidden = 5
no_out = 1

def train_neural_network(X, y, epochs, learning_rate, no_hidden, no_out):
    W1 = np.random.normal(0, 1, (2, no_hidden))
    b1 = np.zeros((no_hidden))
    W2 = np.random.normal(0, 1, (no_hidden, no_out))
    b2 = np.zeros((no_out))

    no_samps = X.shape[0]

    for epoch in range(epochs):
        X, y = shuffle(X, y)

        z1, a1, z2, a2 = forward(X, W1, W2, b1, b2)
        loss = -(y * np.log(a2) + (1 - y) * np.log(1 - a2)).mean()
        accuracy = (np.round(a2) == y).mean()
        print("loss:", loss, "accuracy:", accuracy)

        dw1, db1, dw2, db2 = backward(a1, a2, z1, z2, X, y, no_samps)

        W1 -= dw1 * learning_rate
        b1 -= db1 * learning_rate
        W2 -= dw2 * learning_rate
        b2 -= db2 * learning_rate

        plot_decision(X, W1, W2, b1, b2)

    return W1, b1, W2, b2, accuracy

W1, b1, W2, b2, acc = train_neural_network(X, y, epochs, learning_rate, no_hidden, no_out)
print("\nExercitiul 4\n accuracy:",acc)
