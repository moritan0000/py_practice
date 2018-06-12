import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

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
    train = pd.read_csv(gci_compe_path + "pokemon/train.csv").set_index("id")

    status["Total"] = status["HP"] + status["Attack"] + status["Defense"] + \
                      status["Sp. Atk"] + status["Sp. Def"] + status["Speed"]

    X = pd.DataFrame()
    X["F_total"] = (status.loc[train["First_pokemon"].values, "Total"]).values
    X["S_total"] = (status.loc[train["Second_pokemon"].values, "Total"]).values
    X["F_Legendary"] = 0 + (status.loc[train["First_pokemon"].values, "Legendary"]).values
    X["S_Legendary"] = 0 + (status.loc[train["Second_pokemon"].values, "Legendary"]).values
    X["F_to_S"], X["S_to_F"] = type_calculate(train["First_pokemon"].values, train["Second_pokemon"].values)
    X["HP"] = (status.loc[train["First_pokemon"].values, "HP"]).values
    X["Attack"] = (status.loc[train["First_pokemon"].values, "Attack"]).values
    X["Defense"] = (status.loc[train["Second_pokemon"].values, "Defense"]).values
    X["Sp. Atk"] = (status.loc[train["First_pokemon"].values, "Sp. Atk"]).values
    X["Sp. Def"] = (status.loc[train["First_pokemon"].values, "Sp. Def"]).values
    X["Speed"] = (status.loc[train["First_pokemon"].values, "Speed"]).values
    X = X.apply(lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0))

    y = 0 + (train["Winner"] == train["First_pokemon"])

    test = pd.read_csv(gci_compe_path + "pokemon/test.csv").set_index("id")
    X_test = pd.DataFrame()
    X_test["F_total"] = (status.loc[test["First_pokemon"].values, "Total"]).values
    X_test["S_total"] = (status.loc[test["Second_pokemon"].values, "Total"]).values
    X_test["F_Legendary"] = 0 + (status.loc[test["First_pokemon"].values, "Legendary"]).values
    X_test["S_Legendary"] = 0 + (status.loc[test["Second_pokemon"].values, "Legendary"]).values
    X_test["F_to_S"], X_test["S_to_F"] = type_calculate(test["First_pokemon"].values, test["Second_pokemon"].values)
    X_test["HP"] = (status.loc[test["First_pokemon"].values, "HP"]).values
    X_test["Attack"] = (status.loc[test["First_pokemon"].values, "Attack"]).values
    X_test["Defense"] = (status.loc[test["Second_pokemon"].values, "Defense"]).values
    X_test["Sp. Atk"] = (status.loc[test["First_pokemon"].values, "Sp. Atk"]).values
    X_test["Sp. Def"] = (status.loc[test["First_pokemon"].values, "Sp. Def"]).values
    X_test["Speed"] = (status.loc[test["First_pokemon"].values, "Speed"]).values
    X_test = X_test.apply(lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0))

    (X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = linear_model.LinearRegression()
    clf.fit(X_train, y_train)
    val_result = clf.predict(X_val)
    val_result = (val_result - min(val_result) + 1e-10) / (max(val_result) - min(val_result) + 2e-10)
    logloss_v = -sum(y_val * np.log(val_result) + (1 - y_val) * np.log(1 - val_result)) / len(y_val)
    print("LogLoss:", logloss_v, "If Random:", -(1 * np.log(0.5) + 0 * np.log(0.5)))

    test_result = clf.predict(X_test)
    test_result = (test_result - min(test_result) + 1e-10) / (max(test_result) - min(test_result) + 2e-10)

    out = pd.DataFrame()
    out["id"] = [i for i in range(len(test_result))]
    out["probability"] = test_result
    print(out)

    out.to_csv(gci_compe_path + "pokemon/submission.csv", index=False)


def arrest():
    train_data = pd.read_csv(gci_compe_path + "arrest/train.csv")
    test_data = pd.read_csv(gci_compe_path + "arrest/test.csv")
    # DELETE DUPLICATE
    train_data["search_type_raw"] = train_data["search_type_raw"].map(lambda x: 0 if pd.isnull(x) else x)
    train_data["search_type"] = train_data["search_type"].map(lambda x: 0 if pd.isnull(x) else x)
    train_data["contraband_found"] = train_data["contraband_found"].map(lambda x: 1 if x else 0)
    train_data = train_data.dropna()

    X = train_data[["driver_age_raw", "contraband_found"]]
    X["stop_h"] = train_data["stop_time"].map(lambda x: int(x[:2]) / 24)
    X["d_gen"] = train_data["driver_gender"].map(lambda x: 1 if x == "M" else 0)

    X_dummy = pd.get_dummies(train_data[["driver_race", "search_type", "stop_duration"]], drop_first=True)
    X = pd.merge(X, X_dummy, left_index=True, right_index=True)

    Y = train_data.is_arrested

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=0)

    model_LR = linear_model.LinearRegression()
    model_Log = LogisticRegression()
    model_SVM = LinearSVC()
    model_DT = DecisionTreeClassifier()
    model_kNN = KNeighborsClassifier()
    for model in [model_LR, model_Log, model_SVM, model_DT, model_kNN]:
        model.fit(X_train, y_train)
        print(model.__class__.__name__)
        print(model.score(X_train, y_train))
        print(model.score(X_val, y_val))
        print(roc_auc_score(y_val, model.predict(X_val)))


arrest()
