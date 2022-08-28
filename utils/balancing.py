import pandas as pd
from scipy.stats import chi2_contingency
from constants import CATEGORICAL_COLUMNS


def column_balance(df, column):
    counts = df.groupby(['T', column]).count().id
    percentages = counts.groupby(level=0).apply(lambda x: round(100 * x / float(x.sum()), 2)).copy()
    counts.rename('No.', inplace=True)
    percentages.rename("%", inplace=True)
    balance = pd.concat([counts, percentages], axis=1).reset_index()
    balance = pd.melt(balance, id_vars=['T', column], value_vars=['No.', '%'])
    balance = pd.pivot_table(balance, index=column, columns=['T', 'variable'])
    return balance


def column_balance_prop(df, column):
    percentages = df.groupby(['T', column]).agg({'est': 'sum'}).copy()
    balance = percentages.reset_index()
    balance = pd.pivot_table(balance, index=column, columns=['T'], values='est')
    balance[0] = balance[0] / sum(balance[0])
    balance[1] = balance[1] / sum(balance[1])
    return balance


def chi_square_test(df, column):
    counts = df.groupby(['T', column]).count().id
    counts = pd.DataFrame(counts)
    pivoted = pd.pivot_table(counts, index='T', columns=column)
    chi2, p_value = chi2_contingency(pivoted)[:2]
    res = pd.DataFrame(data=[[round(chi2, 2), p_value]], columns=['Chi Square', 'p value'], index=[column])
    res['p value'] = res['p value'].apply(lambda x: round(x, 3) if x >= 0.05 else '<0.05')
    return res


def chi_square_test_prop(pivoted, column):
    chi2, p_value = chi2_contingency(pivoted)[:2]
    res = pd.DataFrame(data=[[round(chi2, 2), p_value]], columns=['Chi Square', 'p value'], index=[column])
    res['p value'] = res['p value'].apply(lambda x: round(x, 3) if x >= 0.05 else '<0.05')
    return res


def print_balance_table(df, columns=CATEGORICAL_COLUMNS):
    df_list = []
    chi_list = []
    for column in columns:
        df_list.append(column_balance(df, column))
        chi_list.append(chi_square_test(df, column))
    balance = pd.concat(df_list, keys=CATEGORICAL_COLUMNS)
    chi = pd.concat(chi_list, keys=CATEGORICAL_COLUMNS)
    return balance, chi


def print_balance_table_est(df, est, columns=CATEGORICAL_COLUMNS):
    df_list = []
    chi_list = []
    df['est'] = est
    for column in columns:
        pivoted = column_balance_prop(df, column)
        df_list.append(pivoted)
        chi_list.append(chi_square_test_prop(pivoted.T, column))
    balance = pd.concat(df_list, keys=CATEGORICAL_COLUMNS)
    chi = pd.concat(chi_list, keys=CATEGORICAL_COLUMNS)
    return balance, chi
