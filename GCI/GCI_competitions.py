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
    clf = linear_model.LinearRegression()

    sample = pd.read_csv(gci_competitions_path + "wine_quality/train.csv")
    sample_norm = sample.apply(lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0))
    test_data = pd.read_csv(gci_competitions_path + "wine_quality/test.csv")

    predictor_var = sample_norm.drop("quality", axis=1)

    clf.fit(predictor_var.values, sample_norm["quality"])

    print(pd.DataFrame.from_dict({"Name": predictor_var.columns,
                                  "Coefficients": clf.coef_}))
    print("Intercept:", clf.intercept_)


wine_quality_with_normalization()


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
