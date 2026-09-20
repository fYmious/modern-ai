import numpy as np
from sklearn import preprocessing as pre
from sklearn import linear_model
import matplotlib.pyplot as plt
from data.utilities import visualize_classifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from data.utilities import visualize_classifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
import pickle
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle


def price_prediction() -> None:
    data = datasets.fetch_california_housing()

    X, y = shuffle(data.data, data.target, random_state=7)

    X, y = X[:2000], y[:2000]

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    num_training = int(0.8 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    sv_regressor = SVR(kernel="linear", C=1.0, epsilon=0.1)
    sv_regressor.fit(X_train, y_train)

    y_test_pred = sv_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    evs = explained_variance_score(y_test, y_test_pred)
    print("Mean squared error =", mse)
    print("Explained variance score =", evs)

    test_data = [8.3, 41, 6.98, 1.02, 322, 2.55, 37.88, -122.23]
    test_data_scaled = scaler.transform([test_data])
    print("\nPredicted price:", sv_regressor.predict(test_data_scaled)[0])


def multivar_regression() -> None:
    input_file = "data/data_multivar_regr.txt"

    data = np.loadtxt(input_file, delimiter=",")
    X, y = data[:, :-1], data[:, -1]

    num_training = int(0.8 * len(X))
    num_test = len(X) - num_training
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    linear_regressor = linear_model.LinearRegression()
    linear_regressor.fit(X_train, y_train)

    y_test_pred = linear_regressor.predict(X_test)

    print("Linear Regressor performance:")
    print(
        "Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2)
    )
    print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
    print(
        "Median absolute error =",
        round(sm.median_absolute_error(y_test, y_test_pred), 2),
    )
    print(
        "Explained variance score =",
        round(sm.explained_variance_score(y_test, y_test_pred), 2),
    )
    print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

    polynomial = PolynomialFeatures(degree=10)
    X_train_transformed = polynomial.fit_transform(X_train)
    datapoint = [[7.75, 6.35, 5.56]]
    poly_datapoint = polynomial.fit_transform(datapoint)

    poly_linear_model = linear_model.LinearRegression()
    poly_linear_model.fit(X_train_transformed, y_train)
    print("\nLinear regression:\n", linear_regressor.predict(datapoint))
    print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))


def regression() -> None:
    input_file = "data/data_singlevar_regr.txt"

    data = np.loadtxt(input_file, delimiter=",")
    X, y = data[:, :-1], data[:, -1]

    num_training = int(0.8 * len(X))
    num_test = len(X) - num_training
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    regressor = linear_model.LinearRegression()
    regressor.fit(X_train, y_train)

    y_test_pred = regressor.predict(X_test)

    plt.scatter(X_test, y_test, color="green")
    plt.plot(X_test, y_test_pred, color="black", linewidth=4)
    plt.xticks(())
    plt.yticks(())
    plt.savefig("high_res.png", bbox_inches="tight", dpi=300)
    plt.show()

    print("Linear regressor performance:")
    print(
        "Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2)
    )
    print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
    print(
        "Median absolute error =",
        round(sm.median_absolute_error(y_test, y_test_pred), 2),
    )
    print(
        "Explain variance score =",
        round(sm.explained_variance_score(y_test, y_test_pred), 2),
    )
    print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

    output_model_file = "out/model.pkl"

    with open(output_model_file, "wb") as f:
        pickle.dump(regressor, f)

    with open(output_model_file, "rb") as f:
        regressor_model = pickle.load(f)

    y_test_pred_new = regressor_model.predict(X_test)
    print(
        "\nNew mean absolute error =",
        round(sm.mean_absolute_error(y_test, y_test_pred_new), 2),
    )


def svm() -> None:
    input_file = "data/income_data.txt"

    X = []
    y = []
    count_class1 = 0
    count_class2 = 0
    max_datapoints = 25000

    with open(input_file, "r") as f:
        for line in f.readlines():
            if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
                break

            if "?" in line:
                continue

            data = line[:-1].split(", ")
            if data[-1] == "<=50K" and count_class1 < max_datapoints:
                X.append(data)
                count_class1 += 1
            if data[-1] == ">50K" and count_class2 < max_datapoints:
                X.append(data)
                count_class2 += 1

    X = np.array(X)

    label_encoder = []
    X_encoded = np.empty(X.shape)
    for i, item in enumerate(X[0]):
        if item.isdigit():
            X_encoded[:, i] = X[:, i]
        else:
            label_encoder.append(pre.LabelEncoder())
            X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

    X = X_encoded[:, :-1].astype(int)
    y = X_encoded[:, -1].astype(int)

    classifier = OneVsOneClassifier(LinearSVC(random_state=0))

    classifier.fit(X, y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=5
    )
    classifier = OneVsOneClassifier(LinearSVC(random_state=0))
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)

    f1 = model_selection.cross_val_score(classifier, X, y, scoring="f1_weighted", cv=3)
    print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")

    input_data = [
        "23",
        "Private",
        "122272",
        "11th",
        "7",
        "Never-married",
        "Sales",
        "Own-child",
        "Black",
        "Male",
        "0",
        "0",
        "25",
        "United-States",
    ]

    input_data_encoded = [-1] * len(input_data)
    count = 0
    for i, item in enumerate(input_data):
        if item.isdigit():
            input_data_encoded[i] = int(input_data[i])
        else:
            input_data_encoded[i] = int(
                label_encoder[count].transform([input_data[i]])[0]
            )
            count += 1
    input_data_encoded = np.array(input_data_encoded)

    predicted_class = classifier.predict(input_data_encoded.reshape(1, -1))
    print(label_encoder[-1].inverse_transform(predicted_class)[0])


def cmatrix() -> None:
    true_labels = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
    pred_labels = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]

    confusion_mat = confusion_matrix(true_labels, pred_labels)

    plt.imshow(confusion_mat, interpolation="nearest", cmap=plt.cm.gray)
    plt.title("Confusion matrix")
    plt.colorbar()
    ticks = np.arange(5)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")
    # plt.savefig("high_res.png", bbox_inches="tight", dpi=300)
    plt.show()

    targets = ["Class-0", "Class-1", "Class-2", "Class-3", "Class-4"]
    print("\n", classification_report(true_labels, pred_labels, target_names=targets))


def naive_bayes_classifier() -> None:
    input_file = "data/data_multivar_nb.txt"

    data = np.loadtxt(input_file, delimiter=",")
    X, y = data[:, :-1], data[:, -1]

    classifier = GaussianNB()

    classifier.fit(X, y)

    y_pred = classifier.predict(X)

    accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
    print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")

    visualize_classifier(classifier, X, y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=3
    )
    classifier_new = GaussianNB()
    classifier_new.fit(X_train, y_train)
    y_test_pred = classifier_new.predict(X_test)

    accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
    print("Accuracy of the new classifier =", round(accuracy, 2), "%")

    visualize_classifier(classifier_new, X_test, y_test)

    num_folds = 3
    accuracy_values = model_selection.cross_val_score(
        classifier, X, y, scoring="accuracy", cv=num_folds
    )
    print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")
    precision_values = model_selection.cross_val_score(
        classifier, X, y, scoring="precision_weighted", cv=num_folds
    )
    print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")
    recall_values = model_selection.cross_val_score(
        classifier, X, y, scoring="recall_weighted", cv=num_folds
    )
    print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")
    f1_values = model_selection.cross_val_score(
        classifier, X, y, scoring="f1_weighted", cv=num_folds
    )
    print("F1: " + str(round(100 * f1_values.mean(), 2)) + "%")


def logistic_classifier(c: int) -> None:
    X = np.array(
        [
            [3.1, 7.2],
            [4, 6.7],
            [2.9, 8],
            [5.1, 4.5],
            [6, 5],
            [5.6, 5],
            [3.3, 0.4],
            [3.9, 0.9],
            [2.8, 1],
            [0.5, 3.4],
            [1, 4],
            [0.6, 4.9],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    classifier = linear_model.LogisticRegression(solver="lbfgs", C=c)

    classifier.fit(X, y)

    visualize_classifier(classifier, X, y)


def labels_encoding() -> None:
    input_labels = ["red", "black", "red", "green", "black", "yellow", "white"]

    encoder = pre.LabelEncoder()
    encoder.fit(input_labels)

    print("\nLabel mapping:")
    for i, item in enumerate(encoder.classes_):
        print(item, "-->", i)

    test_labels = ["green", "red", "black"]
    encoded_values = encoder.transform(test_labels)
    print("\nLabels =", test_labels)
    print("Encoded values =", list(encoded_values))

    encoded_values = [3, 0, 4, 1]
    decoded_list = encoder.inverse_transform(encoded_values)
    print("\nEncoded values =", encoded_values)
    print("Decoded labels =", list(decoded_list))


def preprocessing() -> None:
    input_data = np.array(
        [[5.1, -2.9, 3.3], [-1.2, 7.8, -6.1], [3.9, 0.4, 2.1], [7.3, -9.9, -4.5]]
    )

    data_binarized = pre.Binarizer(threshold=2.1).transform(input_data)
    print("\nBinarized data:\n", data_binarized)

    print("\nBEFORE:")
    print("Mean =", input_data.mean(axis=0))
    print("Std deviation =", input_data.std(axis=0))

    data_scaled = pre.scale(input_data)
    print("\nAFTER:")
    print("Mean =", data_scaled.mean(axis=0))
    print("Std deviation =", data_scaled.std(axis=0))

    data_scaler_minmax = pre.MinMaxScaler(feature_range=(0, 1))
    data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
    print("\nMin max scaled data:\n", data_scaled_minmax)

    data_normalized_l1 = pre.normalize(input_data, norm="l1")
    data_normalized_l2 = pre.normalize(input_data, norm="l2")
    print("\nL1 normalized data:\n", data_normalized_l1)
    print("\nL2 normalized data:\n", data_normalized_l2)


if __name__ == "__main__":
    # preprocessing()
    # labels_encoding()
    # logistic_classifier(100)
    # naive_bayes_classifier()
    # cmatrix()
    # svm()
    # regression()
    # multivar_regression()
    price_prediction()
