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

x = np.arange(0, np.pi * 2, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sinx, cosx")
plt.legend()
plt.show()
    