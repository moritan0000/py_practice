# Problem No. 1
import numpy as np

x = [[-1],
     [2],
     [-3],
     [4],
     [-5]]

y = [[11],
     [12],
     [13],
     [14],
     [15]]

s1 = [["A"],
      ["B"],
      ["C"],
      ["D"],
      ["E"]]

s2 = [["a"],
      ["b"],
      ["c"],
      ["d"],
      ["e"]]

A = [[1, 2, 3, 4, 5],
     [6, 7, 8, 9, 10],
     [11, 12, 13, 14, 15],
     [16, 17, 18, 19, 20]]


def add_vector(x, y):
    z = [[x[i][0] + y[i][0]] for i in range(len(x))]
    return z


print("add_vector:", all((np.array(x) + np.array(y)) == add_vector(x, y)))


def multiply_matrix_and_vector(A, x):
    B = []
    for i in range(len(A)):
        tmp = 0
        for j in range(len(x)):
            tmp += A[i][j] * x[j][0]
        B.append([tmp])
    return B


print("A dot x:", all(np.dot(A, x) == multiply_matrix_and_vector(A, x)))


def transpose_matrix(A):
    B = []
    for i in range(len(A[0])):
        row = []
        for j in range(len(A)):
            row.append(A[j][i])
        B.append(row)
    return B


print("A.T:", all((np.array(A).T == transpose_matrix(A)).flatten()))


def relu(x):
    z = [[max(0, x[i][0])] for i in range(len(x))]
    return z


print("relu:", all(relu(x) == np.maximum(0, x)))


def softmax(x):
    import math
    denominator = sum([math.exp(x[i][0]) for i in range(len(x))])
    z = [[math.exp(x[i][0]) / denominator] for i in range(len(x))]
    return z


print("softmax:", all(softmax(x) == np.exp(x) / np.sum(np.exp(x))))
