import numpy as np

import scipy as sp
from scipy import stats
from scipy import integrate
import scipy.linalg as linalg

import pandas as pd
from pandas import DataFrame

from sklearn import linear_model
from sklearn.datasets import load_iris

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


url_winequality_data = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def homework3(url_winequality_data):
    df = pd.read_csv(url_winequality_data, sep=";")
    X = df["volatile acidity"].values.reshape(-1, 1)
    y = df["quality"].values.reshape(-1, 1)

    clf = linear_model.LinearRegression()
    clf.fit(X, y)
    my_result = clf.score(X, y)
    return my_result


iris = load_iris()


def homework4(iris):
    print(iris.target_names)
    my_result = 0
    return my_result


# print(homework4(iris))


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
    for customer_id in set(target_online_retail_data_tb["CustomerID"]):
        price_and_id[str(customer_id)[:-2]] = np.sum(
            target_online_retail_data_tb["TotalPrice"][target_online_retail_data_tb["CustomerID"] == customer_id])
    total_price_per_customer = pd.DataFrame(list(price_and_id.items()), columns=["CustomerID", "TotalPrice"])
    total_price_per_customer = total_price_per_customer.sort_values(by="TotalPrice", ascending=False)

    num = len(total_price_per_customer)
    prices = [np.sum(total_price_per_customer["TotalPrice"][num // 10 * i:num // 10 * (i + 1)]) for i in range(10)]
    my_result = (prices / np.sum(prices))[::-1]
    return my_result


# print(homework7(target_online_retail_data_tb))
def homework8():
    my_result = 0
    return my_result
