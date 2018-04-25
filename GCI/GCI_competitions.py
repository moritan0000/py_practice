import pandas as pd
import numpy as np
import sklearn as sl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model

gci_competitions_path = "../../../../weblab/weblab_datascience/competitions/"

# data = pd.read_csv(gci_compe_path + "wine_quality/train.csv")
train_data = np.loadtxt(gci_competitions_path + "wine_quality/train.csv",
                        delimiter=",", skiprows=1, dtype=float)
test_data = np.loadtxt(gci_competitions_path + "wine_quality/test.csv",
                       delimiter=",", skiprows=1, dtype=float)

labels = train_data[:, -1:]
features = train_data[:, :-1]
# features = preprocessing.minmax_scale(data[:, :-1])

x_train, x_test, y_train, y_test \
    = train_test_split(features, np.ravel(labels), test_size=0.2)

clf = linear_model.LinearRegression()

print(clf.fit(features, labels))
