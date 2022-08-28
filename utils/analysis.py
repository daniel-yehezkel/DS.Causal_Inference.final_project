import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display


def printmd(string):
    display(Markdown(string))


def two_classes_analysis(data_a: pd.DataFrame,
                         data_b: pd.DataFrame,
                         numeric_attributes: list):
    """
    make hist/bar per feature

    :param data_a:
    :param data_b:
    :param numeric_attributes:
    :return:
    """
    fig, axes = plt.subplots(7, 5, figsize=(5 * 8, 9 * 5))
    for i in range(7):
        for j in range(5):
            col_name = data_a.columns[i * 5 + j]
            if col_name in numeric_attributes:
                axes[i, j].hist(data_a[col_name], alpha=0.5, label="Math Class")
                axes[i, j].hist(data_b[col_name], alpha=0.5, label="Portuguese Class")
            else:
                # if type(math_data[col_name][0]) == str:
                sm = data_a[col_name].value_counts().sort_index()
                sp = data_b[col_name].value_counts().sort_index()

                br1 = np.arange(len(sm))
                br2 = [x + 0.25 for x in br1]

                axes[i, j].bar(br1, sm.values, width=0.25, label="Math Class")
                axes[i, j].bar(br2, sp.values, width=0.25, label="Portuguese Class")
                axes[i, j].set_xticks([r + 0.125 for r in range(len(sm))])
                axes[i, j].set_xticklabels(list(sm.index))

            axes[i, j].set_xlabel(col_name)
            axes[i, j].legend()
            axes[i, j].set_ylabel("count")
            axes[i, j].set_title(f"{col_name} Histogram")

    fig.savefig("features_hists.png")
    fig.show()
