import numpy as np

import scipy as sp
from scipy import stats, integrate
import scipy.linalg as linalg

import pandas as pd
from pandas import DataFrame

from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# 1, 2, 5, 7 solved in time
# 8, 9, 11, 12, 13 waiting for scoring
# 3, 4, 6 solved late
#  unsolved

input_data = {'key_1': 100,
              'key_2': 100,
              'key_3': 300,
              'key_4': 400,
              'key_5': 500}


def homework1(input_data):
    my_result = sum(input_data.values())
    return my_result


A = np.array([[3, 2, 1],
              [5, 3, 7],
              [1, 1, 1]])


def homework2(A):
    my_result = linalg.det(A)
    return my_result


# url_winequality_data = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def homework3(url_winequality_data):
    df = pd.read_csv(url_winequality_data, sep=";")
    X = df["volatile acidity"].values.reshape(-1, 1)
    y = df["quality"].values.reshape(-1, 1)

    clf = linear_model.LinearRegression()
    clf.fit(X, y)
    my_result = clf.score(X, y)
    return my_result


# iris = load_iris()


def homework4(iris):
    iris_df = pd.DataFrame.from_dict({"target": iris["target"],
                                      "sepal length": iris["data"][:, 0],
                                      "sepal width": iris["data"][:, 1],
                                      "petal length": iris["data"][:, 2],
                                      "petal width": iris["data"][:, 3], })
    setosa = iris_df[iris_df.target == 0]
    versicolor = iris_df[iris_df.target == 1]
    t, p = stats.ttest_rel(setosa["sepal length"], versicolor["sepal length"])
    my_result = p
    return my_result


def homework5():
    my_result = integrate.quad(lambda x: np.exp(-x ** 2), -np.inf, np.inf)[0]
    return my_result


def homework6(url_winequality_data):
    data = pd.read_csv(url_winequality_data, sep=";")
    sul = data["total sulfur dioxide"].sort_values()
    num = len(sul.values) // 5 + 1
    ave = [np.mean(sul[num * i:num * (i + 1)]) for i in range(5)]
    my_result = max(ave) + min(ave)
    return my_result


# print(homework6(url_winequality_data))


# # init part(データの読み込みと前処理)
# file_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
# online_retail_data = pd.ExcelFile(file_url)
# online_retail_data_table = online_retail_data.parse('Online Retail')
#
# online_retail_data_table['cancel_flg'] = online_retail_data_table.InvoiceNo.map(lambda x: str(x)[0])
#
# # 数字があるものとIDがNullでないものが対象
# target_online_retail_data_tb = online_retail_data_table[(online_retail_data_table.cancel_flg == '5')
#                                                         & (online_retail_data_table.CustomerID.notnull())]
#
# target_online_retail_data_tb = target_online_retail_data_tb.assign(
#     TotalPrice=target_online_retail_data_tb.Quantity * target_online_retail_data_tb.UnitPrice)


def homework7(target_online_retail_data_tb):
    price_and_id = {}
    for customer_id in set(target_online_retail_data_tb["CustomerID"]):
        price_and_id[str(customer_id)[:-2]] = np.sum(
            target_online_retail_data_tb["TotalPrice"][target_online_retail_data_tb["CustomerID"] == customer_id])
    total_price_per_customer = pd.DataFrame(list(price_and_id.items()), columns=["CustomerID", "TotalPrice"])
    total_price_per_customer = total_price_per_customer.sort_values(by="TotalPrice", ascending=False)

    num = len(total_price_per_customer)
    prices = [np.sum(total_price_per_customer["TotalPrice"][num // 10 * i:num // 10 * (i + 1)]) for i in range(10)]
    my_result = (prices / np.sum(prices))[::-1]
    return my_result


def homework8():
    my_result = "ZWE"
    return my_result


def homework9():
    my_result = 3367
    return my_result


# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(
#     iris.data, iris.target, stratify=iris.target, random_state=0)
#
# best_score = 0
# best_method = ""


def homework11(X_train, X_test, y_train, y_test, best_score, best_method):
    accuracies = {}
    model_LR = LogisticRegression()
    model_SVM = LinearSVC()
    model_DT = DecisionTreeClassifier()
    model_kNN = KNeighborsClassifier(n_neighbors=6)

    for model in [model_LR, model_SVM, model_DT, model_kNN]:
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        accuracies[model.__class__.__name__] = score
        if score > best_score:
            best_score = score
            best_method = model.__class__.__name__
    my_result = best_method
    return my_result


# print(homework11(X_train, X_test, y_train, y_test, best_score, best_method))

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

file_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
online_retail_data = pd.ExcelFile(file_url)
online_retail_data_table = online_retail_data.parse('Online Retail')

online_retail_data_table['cancel_flg'] = online_retail_data_table.InvoiceNo.map(lambda x: str(x)[0])
target_online_retail_data_tb = online_retail_data_table[(online_retail_data_table.cancel_flg == '5')
                                                        & (online_retail_data_table.CustomerID.notnull())]
target_online_retail_data_tb = target_online_retail_data_tb.assign(
    TotalPrice=target_online_retail_data_tb.Quantity * target_online_retail_data_tb.UnitPrice)


def homework12(target_online_retail_data_tb):
    target_online_retail_data_tb["InvoiceNo"] = target_online_retail_data_tb["InvoiceNo"].astype(str)
    target_online_retail_data_tb["StockCode"] = target_online_retail_data_tb["StockCode"].astype(str)

    online_retail_data_table_sc = target_online_retail_data_tb.groupby("StockCode").size()
    online_retail_data_table_sc = \
        online_retail_data_table_sc[online_retail_data_table_sc > 1000].sort_values(ascending=False)

    # calculate support for each combination. best_support_items = ['20725', '22383']
    item_list = online_retail_data_table_sc.index
    best_support = 0
    best_support_items = []
    for i in range(len(item_list) - 1):
        for j in range(i + 1, len(item_list)):
            first_item = target_online_retail_data_tb[target_online_retail_data_tb["StockCode"] == item_list[i]]
            second_item = target_online_retail_data_tb[target_online_retail_data_tb["StockCode"] == item_list[j]]
            merge_one_second = pd.merge(first_item, second_item, on="InvoiceNo", how="inner")
            support = len(merge_one_second.InvoiceNo.unique()) / len(
                target_online_retail_data_tb.InvoiceNo.unique())
            print("Support between {} & {}:".format(item_list[i], item_list[j]), support)
            if support > best_support:
                best_support = support
                best_support_items = [item_list[i], item_list[j]]
    # print("Best support:", best_support_items, best_support)

    first_item = target_online_retail_data_tb[target_online_retail_data_tb["StockCode"] == best_support_items[0]]
    second_item = target_online_retail_data_tb[target_online_retail_data_tb["StockCode"] == best_support_items[1]]
    merge_one_second = pd.merge(first_item, second_item, on="InvoiceNo", how="inner")
    a = len(merge_one_second.InvoiceNo.unique()) / len(second_item.InvoiceNo.unique())
    b = len(first_item.InvoiceNo.unique()) / len(target_online_retail_data_tb.InvoiceNo.unique())
    my_result = a / b

    return my_result


# print(homework12(target_online_retail_data_tb))

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

wine = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

X = wine.iloc[:, 0:10].as_matrix()
Y = wine['quality'].as_matrix()


def homework13(X, Y):
    model_LR = LogisticRegression()
    model_SVM = LinearSVC()
    model_DT = DecisionTreeClassifier()
    model_kNN = KNeighborsClassifier(n_neighbors=6)
    model_RF = RandomForestClassifier()
    best_score = 0

    for model in [model_LR, model_SVM, model_DT, model_kNN, model_RF]:
        scores = cross_val_score(model, X, Y, cv=10)
        ave_score = np.average(scores)
        # print(model.__class__.__name__, ave_score)
        if ave_score > best_score:
            best_score = ave_score
            best_method = model.__class__.__name__
    my_result = best_score

    return my_result


# print(homework13(X, Y))
