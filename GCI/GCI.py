import numpy as np
import numpy.random as random

import scipy as sp
from scipy import linalg, interpolate, integrate
from scipy.integrate import odeint
from scipy.optimize import minimize, minimize_scalar, fsolve

import pandas as pd
from pandas import Series, DataFrame

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import math, requests, zipfile, io, pymysql
from io import StringIO

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.externals.six import StringIO
from sklearn.datasets import load_breast_cancer, make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

import pydotplus
from IPython.display import Image


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


# ---------- Chapter11 ----------

def chapter_11():
    if 0:  # Linear Regression
        # auto_data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
        # s = requests.get(auto_data_url).content
        # auto_data = pd.read_csv(io.StringIO(s.decode('utf-8')), header=None)
        # auto_data.columns = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors",
        #                      "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height",
        #                      "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore",
        #                      "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg",
        #                      "price"]
        # auto_data.to_csv("../../../../weblab/weblab_datascience/data/auto_data.csv", index=False)
        auto_data = pd.read_csv("../../../../weblab/weblab_datascience/data/auto_data.csv")
        # for col in auto_data.columns:
        #     print(col, sum(auto_data[col].isin(["?"])))

        sub_auto_data = auto_data[["price", "horsepower", "width", "height"]]
        sub_auto_data = sub_auto_data.replace('?', np.nan).dropna()
        sub_auto_data = sub_auto_data.assign(price=pd.to_numeric(sub_auto_data.price))
        sub_auto_data = sub_auto_data.assign(horsepower=pd.to_numeric(sub_auto_data.horsepower))
        # print(sub_auto_data.corr())

        model_linear = linear_model.LinearRegression()
        model_ridge = linear_model.Ridge()

        X = sub_auto_data.drop("price", axis=1)
        Y = sub_auto_data["price"]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

        for model in [model_linear, model_ridge]:
            clf = model.fit(X_train, y_train)
            print("train:", clf.__class__.__name__, clf.score(X_train, y_train))
            print("test:", clf.__class__.__name__, clf.score(X_test, y_test))
            print(pd.DataFrame.from_dict(({"Name": X.columns,
                                           "Coefficients": clf.coef_})).sort_values(by='Coefficients'))
            print(clf.intercept_)

    if 0:  # Logistic Regression
        # adult_data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        # s = requests.get(adult_data_url).content
        # adult_data = pd.read_csv(io.StringIO(s.decode('utf-8')), header=None)
        # adult_data.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        #                       "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        #                       "hours-per-week", "native-country", "flg-50K"]
        # adult_data.to_csv("../../../../weblab/weblab_datascience/data/adult_data.csv", index=False)
        adult_data = pd.read_csv("../../../../weblab/weblab_datascience/data/adult_data.csv")
        adult_data["fin_flg"] = adult_data["flg-50K"].map(lambda x: 1 if x == ' >50K' else 0)

        X = adult_data[["age", "fnlwgt", "education-num", "capital-gain", "capital-loss"]]
        Y = adult_data['fin_flg']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

        model = linear_model.LogisticRegression()

        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        model.fit(X_train_std, y_train)

        print("train result:", model.score(X_train_std, y_train))
        print("test result:", model.score(X_test_std, y_test))
        print(model.coef_)

    if 0:  # Decision tree
        # mush_data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
        # s = requests.get(mush_data_url).content
        # mush_data = pd.read_csv(io.StringIO(s.decode('utf-8')), header=None)
        # mush_data.columns = ["classes", "cap_shape", "cap_surface", "cap_color", "odor", "bruises",
        #                      "gill_attachment", "gill_spacing", "gill_size", "gill_color", "stalk_shape",
        #                      "stalk_root", "stalk_surface_above_ring", "stalk_surface_below_ring",
        #                      "stalk_color_above_ring", "stalk_color_below_ring", "veil_type", "veil_color",
        #                      "ring_number", "ring_type", "spore_print_color", "population", "habitat"]
        # mush_data.to_csv("../../../../weblab/weblab_datascience/data/mush_data.csv", index=False)
        mush_data = pd.read_csv("../../../../weblab/weblab_datascience/data/mush_data.csv")
        # print(mush_data.head())

        mush_data_dummy = pd.get_dummies(mush_data[['gill_color', 'gill_attachment', 'odor', 'cap_color']])
        mush_data_dummy["flg"] = mush_data["classes"].map(lambda x: 1 if x == 'p' else 0)
        # print(mush_data_dummy.head())
        print(mush_data_dummy.groupby("flg")["flg"].count())
        print(mush_data_dummy.groupby(["cap_color_c", "flg"])["flg"].count().unstack())

        print(mush_data_dummy.groupby(["gill_color_b", "flg"])["flg"].count().unstack())

        X = mush_data_dummy.drop("flg", axis=1)
        Y = mush_data_dummy['flg']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=50)

        tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=50)
        tree_model.fit(X_train, y_train)

        print("train:", tree_model.__class__.__name__, tree_model.score(X_train, y_train))
        print("test:", tree_model.__class__.__name__, tree_model.score(X_test, y_test))

        # dot_data = StringIO()
        # tree.export_graphviz(tree_model, out_file=dot_data)
        # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        # Image(graph.create_png())

    if 0:  # k-NN
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = \
            train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

        training_accuracy = []
        test_accuracy = []

        neighbors_settings = range(1, 101)
        for n_neighbors in neighbors_settings:
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            clf.fit(X_train, y_train)

            training_accuracy.append(clf.score(X_train, y_train))

            test_accuracy.append(clf.score(X_test, y_test))

        plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
        plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("n_neighbors")
        plt.legend()
        plt.show()

    if 1:  # SVM
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = \
            train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=50)

        model = LinearSVC()

        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        model.fit(X_train_std, y_train)
        print("train:", model.__class__.__name__, model.score(X_train_std, y_train))
        print("test:", model.__class__.__name__, model.score(X_test_std, y_test))


# ---------- Chapter11 ----------

def chapter_12():
    pd.set_option("display.max_columns", 20)
    if 0:
        if 0:
            X, y = make_blobs(random_state=10)

            kmeans = KMeans(init="random", n_clusters=3)
            kmeans.fit(X)

            y_pre = kmeans.fit_predict(X)

            merge_data = pd.concat([pd.DataFrame(X[:, 0]), pd.DataFrame(X[:, 1]), pd.DataFrame(y_pre)], axis=1)
            merge_data.columns = ["element1", "element2", "cluster"]

            merge_data_cluster0 = merge_data[merge_data["cluster"] == 0]
            merge_data_cluster1 = merge_data[merge_data["cluster"] == 1]
            merge_data_cluster2 = merge_data[merge_data["cluster"] == 2]

            ax = merge_data_cluster0.plot.scatter(x='element1', y='element2', color='red', label='cluster0');
            merge_data_cluster1.plot.scatter(x='element1', y='element2', color='blue', label='cluster1', ax=ax);
            merge_data_cluster2.plot.scatter(x='element1', y='element2', color='green', label='cluster2', ax=ax);
            plt.show()

        # zip_file_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
        # r = requests.get(zip_file_url, stream=True)
        # z = zipfile.ZipFile(io.BytesIO(r.content))
        # z.extractall("bank/")

        banking_c_data = pd.read_csv("bank/bank-full.csv", sep=";")
        banking_c_data.info()
        print(banking_c_data.head())

        banking_c_data_sub = banking_c_data[['age', 'balance', 'campaign', 'previous']]
        sc = StandardScaler()
        sc.fit(banking_c_data_sub)
        banking_c_data_sub_std = sc.transform(banking_c_data_sub)

        X = banking_c_data_sub_std
        kmeans = KMeans(init='random', n_clusters=6, random_state=0)
        kmeans.fit(X)

        labels = kmeans.labels_
        label_data = pd.DataFrame(labels, columns=["cl_nm"])
        label_data_bycl = label_data.groupby("cl_nm").size()
        print(label_data_bycl)

        # plt.bar([0, 1, 2, 3, 4, 5], label_data_bycl.values)
        # plt.ylabel("count")
        # plt.xlabel("cluster num")
        # plt.grid(True)
        # plt.show()

        if 0:  # determine the number of clusters
            dist_list = []
            for i in range(1, 20):
                kmpp = KMeans(n_clusters=i, init='random', n_init=5, max_iter=100, random_state=0)
                kmpp.fit(X)
                dist_list.append(kmpp.inertia_)
            plt.plot(range(1, 20), dist_list, marker='+')
            plt.xlabel("Number of clusters")
            plt.ylabel("Distortion")
            plt.show()

        merge_data = pd.concat([banking_c_data, label_data], axis=1)
        print(merge_data.head(3))
        absences_bins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100]
        qcut_result = pd.cut(merge_data.age, absences_bins, right=False)
        value_qcut_result = pd.value_counts(qcut_result)
        print(value_qcut_result)

        merge_data_age_cl = pd.concat([merge_data.cl_nm, qcut_result], axis=1)
        cluster_num_age_cross_tb = pd.pivot_table(merge_data_age_cl, index=['cl_nm'], columns=['age'],
                                                  aggfunc=lambda x: len(x), fill_value=0)
        print(cluster_num_age_cross_tb)
        # sns.heatmap(cluster_num_age_cross_tb.apply(lambda x: x / x.sum(), axis=1), cmap="Blues")
        # plt.show()

        merge_data_job_cl = merge_data[['cl_nm', 'job']]
        cluster_num_job_cross_tb = pd.pivot_table(merge_data_job_cl, index=['cl_nm'], columns=['job'],
                                                  aggfunc=lambda x: len(x), fill_value=0)
        print(cluster_num_job_cross_tb)
        # sns.heatmap(cluster_num_job_cross_tb.apply(lambda x: x / x.sum(), axis=1), cmap="Reds")
        # plt.show()
    rng_data = np.random.RandomState(1)
    X = np.dot(rng_data.rand(2, 2), rng_data.randn(2, 200)).T
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


chapter_12()
