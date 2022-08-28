import numpy as np
import pandas as pd


def matching(label, propensity, calipher=0.05, replace=True):
    """
    Performs nearest-neighbour matching for a sample of test and control
    observations, based on the propensity scores for each observation.

    :param label: Series that contains the label for each observation.
    :param propensity: Series that contains the propensity score for each observation.
    :param calipher: Bound on distance between observations in terms of propensity score.
    :param replace: Boolean that indicates whether sampling is with (True) or without replacement (False).
    :return: matches
    """
    treated = propensity[label == 1]
    control = propensity[label == 0]

    # Randomly permute in case of sampling without replacement to remove any bias arising from the
    # ordering of the data set
    matching_order = np.random.permutation(label[label == 1].index)
    matches = {}

    for obs in matching_order:
        # Compute the distance between the treatment observation and all candidate controls in terms of
        # propensity score
        distance = abs(treated[obs] - control)

        # Take the closest match
        if distance.min() <= calipher or not calipher:
            matches[obs] = [distance.argmin()]
            # Remove the matched control from the set of candidate controls in case of sampling without replacement
            if not replace:
                control = control.drop(matches[obs])

    return matches


def matching_to_dataframe(match, covariates, remove_duplicates=False):
    """
    Converts a list of matches obtained from matching() to a DataFrame.
    Duplicate rows are controls that where matched multiple times.

    :param match: Dictionary with a list of matched control observations.
    :param covariates: DataFrame that contains the covariates for the observations.
    :param remove_duplicates: Boolean that indicates whether or not to remove duplicate rows from the result.
    If matching with replacement was used you should set this to False
    :return: matching as data frame
    """
    treated = list(match.keys())
    control = [ctrl for matched_list in match.values() for ctrl in matched_list]
    result = pd.concat([covariates.loc[treated], covariates.loc[control]])
    if remove_duplicates:
        return result.groupby(result.index).first()
    else:
        return result
