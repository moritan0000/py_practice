import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x * y)
print(x ** 5)

A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
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


def plot_step():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


def plot_sigmoid():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(y)
    plt.ylim(-0.1, 1.1)
    plt.show()


A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2], [3, 4], [5, 6]])
print("A dot B:\n", np.dot(A, B))
print("B dot A:\n", np.dot(B, A),"\n")

X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6, ]])
Y = np.dot(X, W)
print(Y)
