import numpy as np
import numpy.random as random

import scipy as sp
from scipy import linalg
from scipy.optimize import minimize_scalar

import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt


def prime_number(n):
    nums = np.array([2, 3, 5, 7] + [i for i in range(11, n, 2) if i % 3 > 0 and i % 5 > 0 and i % 7 > 0])
    ans = np.array([], dtype=int)

    for val in nums:
        mod = val % nums
        if np.sum(mod == 0) == 1:
            ans = np.append(ans, val)
    return ans


# ---------- Chapter2 ----------


random.seed(0)
a = random.randn(16).reshape(4, 4) * 10
print(linalg.det(a))
print(linalg.inv(a))
print(np.dot(a, linalg.inv(a)))
[eig_value, eig_vector] = linalg.eig(a)
print(eig_value)
print(eig_vector)


def sample_function(x):
    return x ** 2 + 2 * x + 1


print(sp.optimize.newton(sample_function, 0))
print(minimize_scalar(sample_function, method="Brent"))

sample_pandas_data = Series.from_array([12, 23, 34, 45, 56, 67, 78, 89, 90, 121],
                                       index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
print(sample_pandas_data)

attri_data1 = {'ID': ['100', '101', '102', '103', '104'],
               'city': ['Tokyo', 'Osaka', 'Kyoto', 'Hokkaidao', 'Tokyo'],
               'birth_year': [1990, 1989, 1992, 1997, 1982],
               'name': ['Hiroshi', 'Akiko', 'Yuki', 'Satoru', 'Steeve']}

attri_data_frame1 = DataFrame(attri_data1, index=["a", "b", "c", "d", "e"])
print(attri_data_frame1)
print(attri_data_frame1[["ID", "city"]].T)
print(attri_data_frame1[attri_data_frame1['city'].isin(["Tokyo", "Osaka"])])
print(attri_data_frame1.drop(['birth_year'], axis=1))

attri_data2 = {'ID': ['100', '101', '102', '105', '107'],
               'math': [50, 43, 33, 76, 98],
               'English': [90, 30, 20, 50, 30],
               'sex': ['M', 'F', 'F', 'M', 'M']}

attri_data_frame2 = DataFrame.from_dict(attri_data2)
print(attri_data_frame2)

print(pd.merge(attri_data_frame1, attri_data_frame2, "outer"))
print(attri_data_frame2.groupby("sex")["math"].mean())


def scatter():
    # 散布図
    random.seed(0)
    x = np.random.randn(300)
    y = np.sin(x) + np.random.randn(300)

    plt.plot(x, y, "o")
    # plt.scatter(x, y)

    plt.title("Title Name")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


def continuous():
    # 連続曲線
    np.random.seed(0)
    numpy_data_x = np.arange(1000)

    numpy_random_data_y = np.random.randn(1000).cumsum()

    plt.plot(numpy_data_x, numpy_random_data_y, label="Label")
    plt.legend()

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


def sin_and_cos():
    plt.subplot(2, 2, 1)
    x1 = np.linspace(-10, 10, 100)
    plt.plot(x1, np.sin(x1))

    plt.subplot(2, 2, 2)
    x2 = np.linspace(-10, 10, 100)
    plt.plot(x2, np.sin(2 * x2))

    plt.subplot(2, 2, 3)
    x3 = np.linspace(-10, 10, 100)
    plt.plot(x3, np.sin(x3))

    plt.subplot(2, 2, 4)
    x4 = np.linspace(-10, 10, 100)
    plt.plot(x4, np.sin(2 * x4))

    plt.grid(True)
    plt.show()


def hist():
    random.seed(0)
    plt.subplot(3, 1, 1)
    plt.hist(np.random.randn(10 ** 5) * 10 + 50, bins=60, range=(20, 80))

    plt.subplot(3, 1, 2)
    plt.hist(random.uniform(0.0, 1.0, 1000), bins=100)

    plt.subplot(3, 1, 3)
    plt.hist(random.uniform(0.0, 1.0, 1000), bins=100)

    plt.grid(True)
    plt.show()


def monte_carlo():
    random.seed(0)
    n = 1000000
    x = random.uniform(-1.0, 1.0, n)
    y = random.uniform(-1.0, 1.0, n)

    r = np.sqrt(x ** 2 + y ** 2)
    mask = r < 1
    print("pi =", np.sum(mask) * 4 / n)

    plt.subplot(2, 1, 1)
    plt.scatter(x[mask], y[mask])

    plt.subplot(2, 1, 2)
    plt.scatter(x[mask == 0], y[mask == 0])

    plt.show()

# ---------- Chapter3 ----------
