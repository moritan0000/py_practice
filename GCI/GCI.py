import numpy as np
import numpy.random as random

import scipy as sp
import scipy.linalg as linalg
from scipy.optimize import minimize_scalar
from scipy import integrate
from scipy import stats

import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.datasets import load_iris


def prime_number(n):
    import numpy as np

    nums = np.array([2, 3, 5, 7] + [i for i in range(11, n, 2) if i % 3 > 0 and i % 5 > 0 and i % 7 > 0])
    ans = np.array([], dtype=int)

    for val in nums:
        mod = val % nums
        if np.sum(mod == 0) == 1:
            ans = np.append(ans, val)

    return ans


input_data = {'key_1': 100,
              'key_2': 100,
              'key_3': 300,
              'key_4': 400,
              'key_5': 500}


def homework1(input_data):
    my_result = sum(input_data.values())
    return my_result


# ---------- Chapter2 ----------

random.seed(0)
a = random.randn(16).reshape(4, 4) * 10
print(linalg.det(a))
print(linalg.inv(a))
print(np.dot(a, linalg.inv(a)))
eig_value, eig_vector = linalg.eig(a)
print(eig_value)
print(eig_vector)


def sample_function(x):
    return x ** 2 + 2 * x + 1


print(sp.optimize.newton(sample_function, 0))
print(minimize_scalar(sample_function, method="Brent"))

sample_pandas_data = pd.Series([12, 23, 34, 45, 56, 67, 78, 89, 90, 121],
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

attri_data_frame2 = DataFrame(attri_data2)
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


def renzoku():
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


A = np.array([[3, 2, 1],
              [5, 3, 7],
              [1, 1, 1]])


def homework2(A):
    my_result = linalg.det(A)
    return my_result


# ---------- Chapter3 ----------

url_winequality_data = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def homework3(url_winequality_data):
    df = pd.read_csv(url_winequality_data, sep=";")
    predictor = df["volatile acidity"].values.reshape(-1, 1)
    objective = df["quality"].values.reshape(-1, 1)

    clf = linear_model.LinearRegression()
    clf.fit(predictor, objective)
    my_result = clf.score(predictor, objective)
    return my_result


iris = load_iris()


def homework4(iris):
    print(iris.target_names)
    my_result = 0
    return my_result


print(homework4(iris))


def homework5():
    my_result = integrate.quad(lambda x: np.exp(-x ** 2), -np.inf, np.inf)[0]
    return my_result


def homework6(url_winequality_data):
    data = pd.read_csv(url_winequality_data, sep=";")
    sul = data["total sulfur dioxide"]
    num = len(sul.values) // 5
    ave = [np.mean(sul[num * i:num * (i + 1)]) for i in range(5)]
    my_result = max(ave) + min(ave)
    return my_result


# init part(データの読み込みと前処理)
file_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
online_retail_data = pd.ExcelFile(file_url)
online_retail_data_table = online_retail_data.parse('Online Retail')

online_retail_data_table['cancel_flg'] = online_retail_data_table.InvoiceNo.map(lambda x: str(x)[0])

# 数字があるものとIDがNullでないものが対象
target_online_retail_data_tb = online_retail_data_table[(online_retail_data_table.cancel_flg == '5')
                                                        & (online_retail_data_table.CustomerID.notnull())]

target_online_retail_data_tb = target_online_retail_data_tb.assign(
    TotalPrice=target_online_retail_data_tb.Quantity * target_online_retail_data_tb.UnitPrice)


def homework7(target_online_retail_data_tb):
    price_and_id = {}
    for id in set(target_online_retail_data_tb["CustomerID"]):
        price_and_id[str(id)[:-2]] = np.sum(
            target_online_retail_data_tb["TotalPrice"][target_online_retail_data_tb["CustomerID"] == id])
    total_price_per_customer = pd.DataFrame(list(price_and_id.items()), columns=["CustomerID", "TotalPrice"])
    total_price_per_customer = total_price_per_customer.sort_values(by="TotalPrice", ascending=False)

    num = len(total_price_per_customer)
    prices = [np.sum(total_price_per_customer["TotalPrice"][num // 10 * i:num // 10 * (i + 1)]) for i in range(10)]
    my_result = (prices / np.sum(prices))[::-1]
    return my_result

# print(homework7(target_online_retail_data_tb))
