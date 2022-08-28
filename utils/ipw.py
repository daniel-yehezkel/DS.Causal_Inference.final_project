from utils.analysis import printmd


def estimate_ate_ipw(T, Y, e):
    n = len(T)
    left_argument = sum(T * Y / e)
    right_argument = sum(((1 - T) * Y) / (1 - e))
    return (left_argument - right_argument) / n


def calc_bootstap_confidence_interval_ipw(df, e):
    deltas = []
    B = 400
    alpha = 0.05
    for i in range(B):  # B=400 as defined earlier

        temp_sample = df.sample(n=200, replace=True)  # allow selecting each row more than once
        temp_treatment = temp_sample["T"]
        temp_Y = temp_sample["Y"]
        temp_e = e[temp_sample.index]
        deltas.append(estimate_ate_ipw(temp_treatment, temp_Y, temp_e))

    deltas.sort()
    CI_3 = [deltas[int(B * (alpha / 2))], deltas[int(B * (1 - alpha / 2))]]
    printmd("**The CI for delta is:**")
    print(CI_3)
    return CI_3
