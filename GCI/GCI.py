import numpy as np
import numpy.random as random

import scipy as sp
import scipy.stats
from scipy import linalg, interpolate, integrate
from scipy.integrate import odeint
from scipy.optimize import minimize, minimize_scalar, fsolve

import pandas as pd
from pandas import Series, DataFrame

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import math

from sklearn import linear_model

import requests, zipfile
from io import StringIO
import io

import pymysql, pymysql.cursors, mysql.connector


# ---------- Chapter1 ----------

def prime_number(n):
    nums = np.array([2, 3, 5, 7] + [i for i in range(11, n + 1, 2) if i % 3 > 0 and i % 5 > 0 and i % 7 > 0])
    ans = np.array([], dtype=int)

    for val in nums:
        mod = val % nums
        if np.sum(mod == 0) == 1:
            ans = np.append(ans, val)
    return ans


# ---------- Chapter2 ----------

def chapter_2():
    random.seed(0)
    a = random.randn(16).reshape(4, 4) * 10
    print(linalg.det(a))
    print(linalg.inv(a))
    print(np.dot(a, linalg.inv(a)))
    [eig_value, eig_vector] = linalg.eig(a)
    print(eig_value)
    print(eig_vector)

    def sample_func(x):
        return x ** 2 + 2 * x + 1

    print(sp.optimize.newton(sample_func, 0))
    print(minimize_scalar(sample_func, method="Brent"))

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

def chapter_3():
    # zip_file_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00356/student.zip"
    # r = requests.get(zip_file_url, stream=True)
    # z = zipfile.ZipFile(io.BytesIO(r.content))
    # z.extractall("student/")

    stu_data_math = pd.read_csv("student/student-mat.csv", sep=";")
    # print(stu_data_math.head())
    # print(stu_data_math.info())

    # plt.hist(stu_data_math["absences"])
    # plt.ylabel("count")
    # plt.xlabel("absences")
    # plt.grid(True)
    # plt.show()

    print(stu_data_math.describe())

    # plt.subplot(1, 2, 1)
    # plt.boxplot([stu_data_math.G1, stu_data_math.G2, stu_data_math.G3])
    # plt.grid(True)
    # plt.subplot(1, 2, 2)
    # plt.boxplot(stu_data_math.absences)
    # plt.grid(True)
    # plt.show()

    print("Coefficient of Values", "\n", stu_data_math.std() / stu_data_math.mean())

    # plt.plot(stu_data_math.G1, stu_data_math.G3, 'o')
    # plt.ylabel("G3 grade")
    # plt.xlabel("G1 grade")
    # plt.grid(True)
    # plt.show()

    # print("Covariance between G1-G3", "\n", np.cov(stu_data_math["G1"], stu_data_math["G3"]))
    # print(sp.stats.pearsonr(stu_data_math["G1"], stu_data_math["G3"]))
    # print("Corr-coef. between G1-G3", "\n", np.corrcoef(stu_data_math["G1"], stu_data_math["G3"]))
    # sns.pairplot(stu_data_math[["Dalc", "Walc", "G1", "G3"]])
    # plt.grid(True)
    # plt.show()

    print(stu_data_math.groupby("Walc")["G1"].mean())

    stu_por = pd.read_csv("student/student-por.csv", sep=";")
    print(stu_por.describe())

    X = stu_data_math.loc[:, ["G1"]].as_matrix()
    Y = stu_data_math['G3'].as_matrix()

    clf = linear_model.LinearRegression()
    clf.fit(X, Y)

    print("回帰係数:", clf.coef_, "切片:", clf.intercept_, "決定係数:", clf.score(X, Y))

    plt.scatter(X, Y)
    plt.plot(X, clf.predict(X))
    plt.xlabel("G1 grade")
    plt.ylabel("G3 grade")
    plt.grid(True)
    plt.show()


# ---------- Chapter5 ----------

def chapter_5():
    # x = np.linspace(0, 10, num=11, endpoint=True)
    # y = np.cos(-x ** 2 / 5.0)
    # f = interpolate.interp1d(x, y, "linear")
    # f2 = interpolate.interp1d(x, y, "cubic")
    # xnew = np.linspace(0, 10, num=30, endpoint=True)
    # plt.plot(x, y, "o", xnew, f(xnew), "-", xnew, f2(xnew), "--")
    # plt.legend(["data", "linear", "cubic"], loc="best")
    # plt.grid(True)
    # plt.show()

    A = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    U, s, Vs = sp.linalg.svd(A)
    m, n = A.shape
    S = sp.linalg.diagsvd(s, m, n)
    print("U.S.V = \n", U @ S @ Vs)

    A = np.identity(5)
    A[0, :] = 1
    A[:, 0] = 1
    A[0, 0] = 5
    b = np.ones(5)
    (LU, piv) = sp.linalg.lu_factor(A)
    L = np.identity(5) + np.tril(LU, -1)
    U = np.triu(LU)
    P = np.identity(5)[piv]
    x = sp.linalg.lu_solve((LU, piv), b)
    print(x)

    print(integrate.quad(lambda x: 4 / (1 + x ** 2), 0, 1))
    print(integrate.dblquad(lambda t, x: np.exp(-x * t) / t ** 4, 0, np.inf, lambda x: 1, lambda x: np.inf))

    def lorentz_func(v, t, p, r, b):
        return [-p * v[0] + p * v[1], -v[0] * v[2] + r * v[0] - v[1], v[0] * v[1] - b * v[2]]

    # p, r, b = 10, 28, 8 / 3
    # v0 = [0.1, 0.1, 0.1]
    # t = np.arange(0, 100, 0.01)
    # v = odeint(lorentz_func, v0, t, args=(p, r, b))
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(v[:, 0], v[:, 1], v[:, 2])
    # plt.title("Lorenz")
    # plt.grid(True)
    # plt.show()

    print(fsolve(lambda x: 2 * x ** 2 + 2 * x - 10, 1), fsolve(lambda x: 2 * x ** 2 + 2 * x - 10, -2))

    x0 = [1, 5, 5, 1]

    def objective(x): return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def constraint1(x): return x[0] * x[1] * x[2] * x[3] - 25.0

    def constraint2(x): return 40 - np.sum(x ** 2)

    b = (1.0, 5.0)
    bnds = (b, b, b, b)
    con1 = {'type': 'ineq', 'fun': constraint1}
    con2 = {'type': 'ineq', 'fun': constraint2}
    cons = [con1, con2]
    sol = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
    print(sol)


# ---------- Chapter9 ----------
def chapter_9():
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='nugiba8256SQ;',
        db='TEST1',
        charset='utf8',
        cursorclass=pymysql.cursors.DictCursor)  # カーソルのクラスを指定できます。ここでは辞書型にしています。

    stmt = conn.cursor()
    sql = "select * from meibo limit 10"
    stmt.execute(sql)
    rows = stmt.fetchall()

    for row in rows:
        print(row['id'], row['name'], row['age'], row['class'], row['height'])

    stmt.close()
    conn.close()

# chapter_9()
