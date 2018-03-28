import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x * y)
print(x ** 5)

A = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]])
B = np.array([10, 20])
print(A * B)
print(A.shape, A.dtype)
print(A[1] ** 4)
for row in A:
    print(row)
A = A.flatten()
print(A[np.array([0, 2, 4, 6])])
print(A > 4)
print(A[A > 4])


def plot_sincos():
    x = np.arange(0, np.pi * 2, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x ** 2)
    y4 = np.cos(2 ** x)
    plt.plot(x, y1, label="sinx")
    plt.plot(x, y2, linestyle="--", label="cosx")
    plt.plot(x, y3, label="sin(x^2)")
    plt.plot(x, y4, label="cos(2^x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sin, cos")
    plt.legend()
    plt.show()


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


print("AND :", AND(0, 0), AND(1, 0), AND(0, 1), AND(1, 1))
print("NAND:", NAND(0, 0), NAND(1, 0), NAND(0, 1), NAND(1, 1))
print("OR  :", OR(0, 0), OR(1, 0), OR(0, 1), OR(1, 1))
print("XOR :", XOR(0, 0), XOR(1, 0), XOR(0, 1), XOR(1, 1))


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def plot_step_and_sigmoid():
    x = np.arange(-10.0, 10.0, 0.1)
    y1 = step_function(x)
    y2 = sigmoid(x)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.ylim(-0.1, 1.1)
    plt.show()


A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print("A dot B:\n", np.dot(A, B))
print("B dot A:\n", np.dot(B, A), "\n")


def identity_function(x):
    return x


def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5],
                              [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4],
                              [0.2, 0.5],
                              [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3],
                              [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y, "\n")


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y, "SUM =", np.sum(y), "\n")

from dlfs.dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

img = x_train[0]
label = t_train[0]
print("label[0]:", label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

import os, pickle


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    name = os.path.dirname(os.path.abspath(__name__))
    joined_path = os.path.join(name, 'dlfs/ch03/sample_weight.pkl')
    data_path = os.path.normpath(joined_path)
    with open(data_path, "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = sigmoid(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i + batch_size])

print("Accuracy:", str(float(accuracy_cnt) / len(x)))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y1 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print(mean_squared_error(y1, t))
print(mean_squared_error(y2, t))
print(cross_entropy_error(y1, t))
print(cross_entropy_error(y2, t))

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape, t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / 2 * h


def function_1(x):
    return 0.02 * x ** 2 + 0.1 * x


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h  # f(x+h)
        fxh1 = f(x)
        x[idx] = tmp_val - h  # f(x-h)
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


print(numerical_gradient(function_2, np.array([3.0, 4.0])))


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


init_x = np.array([-3.0, 4.0])
lr = 0.1
step_num = 100
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
print(x)

plt.plot([-5, 5], [0, 0], '--b')
plt.plot([0, 0], [-5, 5], '--b')
plt.plot(x_history[:, 0], x_history[:, 1], 'o')
plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
