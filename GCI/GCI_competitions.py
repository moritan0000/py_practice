import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import tensorflow as tf

gci_competitions_path = "../../../../weblab/weblab_datascience/competitions/"


def wine_quality_with_normalization():
    train = pd.read_csv(gci_competitions_path + "wine_quality/train.csv")
    train_predictor = train.drop("quality", axis=1)
    train_predictor_norm = train_predictor.apply(lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0))

    test = pd.read_csv(gci_competitions_path + "wine_quality/test.csv")
    test_predictor = test.drop("id", axis=1)
    test_predictor_norm = test_predictor.apply(lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0))

    clf = linear_model.LinearRegression()
    clf.fit(train_predictor_norm.values, train["quality"])

    test_result = clf.predict(test_predictor_norm.values)
    out = pd.DataFrame.from_dict({"quality": test_result})

    print(pd.DataFrame.from_dict({"Name": train_predictor.columns,
                                  "Coefficients": clf.coef_}))
    print("Intercept:", clf.intercept_)
    print(out)
    out.to_csv(gci_competitions_path + "wine_quality/submit.csv", index=False)


def wine_quality_wo_norm():
    train_data = np.loadtxt(gci_competitions_path + "wine_quality/train.csv",
                            delimiter=",", skiprows=1, dtype=float)
    test_data = np.loadtxt(gci_competitions_path + "wine_quality/test.csv",
                           delimiter=",", skiprows=1, dtype=float)

    labels = train_data[:, -1:]
    features = train_data[:, :-1]
    x_train, x_test, y_train, y_test \
        = train_test_split(features, np.ravel(labels), test_size=0.2)
    # features = preprocessing.minmax_scale(data[:, :-1])
