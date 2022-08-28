import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LinearRegression
from sklearn.calibration import calibration_curve
from constants import DISTINCT_STUDENT_ATTRIBUTES


def linear_regression_comparison(data_a: pd.DataFrame,
                                 data_b: pd.DataFrame):
    """
    perform linear regression comparison
    :param data_a: first group
    :param data_b: second group
    """
    gs = data_a.merge(data_b, on=DISTINCT_STUDENT_ATTRIBUTES)[['Y_x', 'Y_y']]
    X = gs["Y_x"].values.reshape(-1, 1)
    Y = gs["Y_y"].values.reshape(-1, 1)
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, Y)
    Y_pred = linear_regressor.predict(X)

    print("intercept", linear_regressor.intercept_)
    print("coef", linear_regressor.coef_)
    print("R^2", linear_regressor.score(X, Y))
    print("R^2_adj", (1 - linear_regressor.score(X, Y) ** 2) * (len(X) - 1) / (len(X) - 1 - 1))

    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.title("Linear regression plot between classes grades")
    plt.xlabel("Math Grade")
    plt.ylabel("Portuguese Grade")
    plt.legend(["linear regression line", "student grades"])
    plt.savefig('Linear regression plot between classes grades.png')
    plt.show()


def plot_propensity_graph(method_name, treatment, propensity_scores):
    """

    :param method_name:
    :param treatment:
    :param propensity_scores:
    :return:
    """
    plt.hist(propensity_scores[treatment == 1], fc=(0, 0, 1, 0.5), bins=20, label='Treated')
    plt.hist(propensity_scores[treatment == 0], fc=(1, 0, 0, 0.5), bins=20, label='Control')
    plt.title(method_name + " propensity scores overlap")
    plt.legend()
    plt.xlabel('propensity score')
    plt.ylabel("number of units")


def plot_calibration_curve(est_dict, T):
    """Plot calibration curve for est w/o and with calibration.
    Taken from sklearn docs"""

    plt.figure(1, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name, est in est_dict.items():
        y_pred = est

        clf_score = brier_score_loss(T, y_pred)
        print("%s:" % name)
        print("\tBrier: %1.3f" % clf_score)

        fraction_of_positives, mean_predicted_value = calibration_curve(T, y_pred, n_bins=10)

        ax1.plot(mean_predicted_value,
                 fraction_of_positives,
                 "s-",
                 label="%s (%1.3f)" % (name, clf_score))

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    ax1.set_xlabel("Mean predicted value")

    plt.tight_layout()
