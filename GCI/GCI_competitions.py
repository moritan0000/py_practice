import pandas as pd
import numpy as np
import sklearn as sl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model

gci_competitions_path = "../../../../weblab/weblab_datascience/competitions/"


def wine_quality_1():
    clf = linear_model.LinearRegression()

    data = pd.read_csv(gci_competitions_path + "wine_quality/train.csv")
    data_norm = data.apply(lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0))

    predictor_var = data_norm.drop("quality", axis=1)
    X = predictor_var.values
    Y = data_norm["quality"].values

    clf.fit(X, Y)

    print(pd.DataFrame({"Name": predictor_var.columns,
                        "Coefficients": clf.coef_}))
    print("Intercept:", clf.intercept_)


wine_quality_1()


def wine_quality_2():
    train_data = np.loadtxt(gci_competitions_path + "wine_quality/train.csv",
                            delimiter=",", skiprows=1, dtype=float)
    test_data = np.loadtxt(gci_competitions_path + "wine_quality/test.csv",
                           delimiter=",", skiprows=1, dtype=float)

    labels = train_data[:, -1:]
    features = train_data[:, :-1]
    x_train, x_test, y_train, y_test \
        = train_test_split(features, np.ravel(labels), test_size=0.2)
    # features = preprocessing.minmax_scale(data[:, :-1])
