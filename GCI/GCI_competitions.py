import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import math

gci_compe_path = "../../../../weblab/weblab_datascience/competitions/"


def wine_quality():
    train = pd.read_csv(gci_compe_path + "wine_quality/train.csv")
    train_predictor = train.drop("quality", axis=1)
    train_predictor_norm = train_predictor.apply(lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0))

    test = pd.read_csv(gci_compe_path + "wine_quality/test.csv")
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
    out.to_csv(gci_compe_path + "wine_quality/submit.csv", index=False)


# def wine_quality_wo_norm():
#     train_data = np.loadtxt(gci_competitions_path + "wine_quality/train.csv",
#                             delimiter=",", skiprows=1, dtype=float)
#     test_data = np.loadtxt(gci_competitions_path + "wine_quality/test.csv",
#                            delimiter=",", skiprows=1, dtype=float)
#
#     labels = train_data[:, -1:]
#     features = train_data[:, :-1]
#     x_train, x_test, y_train, y_test \
#         = train_test_split(features, np.ravel(labels), test_size=0.2)
#     features = preprocessing.minmax_scale(data[:, :-1])


def pokemon():
    def type_calculate(first_ids, second_ids):
        num = len(first_ids)
        f_type_list = (status.loc[first_ids, ["Type 1", "Type 2"]]).values
        s_type_list = (status.loc[second_ids, ["Type 1", "Type 2"]]).values

        f_to_s_list = np.zeros(num)
        s_to_f_list = np.zeros(num)
        for i in range(num):
            f_types = f_type_list[i]
            s_types = s_type_list[i]
            f_to_s = s_to_f = 0
            div1 = div2 = 2
            for f_t in f_types:
                if type(f_t) != str:
                    div1 = 1
                    break
                tmp = 1
                for s_t in s_types:
                    if type(s_t) != str:
                        break
                    tmp *= types.loc[f_t, s_t]
                f_to_s += tmp

            for s_t in s_types:
                if type(s_t) != str:
                    div2 = 1
                    break
                tmp = 1
                for f_t in f_types:
                    if type(f_t) != str:
                        break
                    tmp *= types.loc[s_t, f_t]
                s_to_f += tmp

            f_to_s /= div1
            s_to_f /= div2
            f_to_s_list[i] = f_to_s
            s_to_f_list[i] = s_to_f
        return f_to_s_list, s_to_f_list

    types = pd.read_csv(gci_compe_path + "pokemon/types.csv").set_index("Attack\Defense")

    status = pd.read_csv(gci_compe_path + "pokemon/pokemon.csv").set_index("#")
    status["Total"] = status["HP"] + status["Attack"] + status["Defense"] + \
                      status["Sp. Atk"] + status["Sp. Def"] + status["Speed"]

    train = pd.read_csv(gci_compe_path + "pokemon/train.csv").set_index("id")
    train["First_is_winner"] = 0 + (train["Winner"] == train["First_pokemon"])
    train["F_total"] = (status.loc[train["First_pokemon"].values, "Total"]).values
    train["S_total"] = (status.loc[train["Second_pokemon"].values, "Total"]).values
    train["F_to_S"], train["S_to_F"] = type_calculate(train["First_pokemon"].values, train["Second_pokemon"].values)

    test = pd.read_csv(gci_compe_path + "pokemon/test.csv").set_index("id")

    # (X_train, X_validate, y_train, y_validate) = train_test_split(X, y, test_size=0.2, random_state=0)
    print(train)


pokemon()
