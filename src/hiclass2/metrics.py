"""Helper functions to compute hierarchical evaluation metrics."""
import numpy as np
from sklearn.utils import check_array
import pandas as pd

from hiclass2.HierarchicalClassifier import make_leveled
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



def _validate_input(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    y_pred = make_leveled(y_pred)
    y_true = make_leveled(y_true)
    if y_true.ndim == 2:
        # print("yay")
        y_true = check_array(y_true, dtype=None)
    else:
        try:
            if isinstance(y_true, pd.Series):
                y_true = y_true.to_numpy().reshape(-1, 1)
            if isinstance(y_true, np.ndarray):
                y_true = y_true.reshape(-1, 1)
            # print("true", type(y_true), y_true.shape)
            y_true = check_array(y_true, dtype=None)
        except Exception as e:
            print("Error 19! ", e)
    if y_pred.ndim == 2:
        # print("yay2")
        y_pred = check_array(y_pred, dtype=None)
    else:
        try:
            if isinstance(y_pred, pd.Series):
                y_pred = y_pred.to_numpy().reshape(-1, 1)
            if isinstance(y_pred, np.ndarray):
                y_pred = y_pred.reshape(-1, 1)
            # print("pred", type(y_pred), y_pred.shape)
            y_pred = check_array(y_pred, dtype=None)
        except Exception as e:
            print("Error 26! ", e)

    # cm = confusion_matrix(y_true, y_pred)
    # cm = confusion_matrix(y_true, y_pred, labels=clf.classes_)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    # disp.plot()
    # plt.matshow(cm)
    # plt.savefig('test.png')
    # print(confusion_matrix(y_true, y_pred))
    return y_true, y_pred

# def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
#     cm = confusion_matrix(y_true, y_pred)
#     # cm = confusion_matrix(y_true, y_pred, labels=clf.classes_)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
#     disp.plot()
#     plt.savefig('test.png')
#     return 1


def precision(y_true: np.ndarray, y_pred: np.ndarray):
    r"""
    Compute precision score for hierarchical classification.

    :math:`hP = \displaystyle{\frac{\sum_{i}| \alpha_i \cap \beta_i |}{\sum_{i}| \alpha_i |}}`,
    where :math:`\alpha_i` is the set consisting of the most specific classes predicted
    for test example :math:`i` and all their ancestor classes, while :math:`\beta_i` is the
    set containing the true most specific classes of test example :math:`i` and all
    their ancestors, with summations computed over all test examples.

    Parameters
    ----------
    y_true : np.array of shape (n_samples, n_levels)
        Ground truth (correct) labels.
    y_pred : np.array of shape (n_samples, n_levels)
        Predicted labels, as returned by a classifier.
    Returns
    -------
    precision : float
        What proportion of positive identifications was actually correct?
    """
    y_true, y_pred = _validate_input(y_true, y_pred)
    y_pred = y_pred.astype(int)
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        ground_truth_set = set(ground_truth)
        ground_truth_set.discard("")
        prediction_set = set(prediction)
        prediction_set.discard("")
        sum_intersection = sum_intersection + len(
            ground_truth_set.intersection(prediction_set)
        )
        sum_prediction_and_ancestors = sum_prediction_and_ancestors + len(
            prediction_set
        )
    precision = sum_intersection / sum_prediction_and_ancestors
    # print(confusion_matrix(y_true, y_pred))
    return precision


def recall(y_true: np.ndarray, y_pred: np.ndarray):
    r"""
    Compute recall score for hierarchical classification.

    :math:`\displaystyle{hR = \frac{\sum_i|\alpha_i \cap \beta_i|}{\sum_i|\beta_i|}}`,
    where :math:`\alpha_i` is the set consisting of the most specific classes predicted
    for test example :math:`i` and all their ancestor classes, while :math:`\beta_i` is the
    set containing the true most specific classes of test example :math:`i` and all
    their ancestors, with summations computed over all test examples.

    Parameters
    ----------
    y_true : np.array of shape (n_samples, n_levels)
        Ground truth (correct) labels.
    y_pred : np.array of shape (n_samples, n_levels)
        Predicted labels, as returned by a classifier.
    Returns
    -------
    recall : float
        What proportion of actual positives was identified correctly?
    """
    y_true, y_pred = _validate_input(y_true, y_pred)
    y_pred = y_pred.astype(int)
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        ground_truth_set = set(ground_truth)
        ground_truth_set.discard("")
        prediction_set = set(prediction)
        prediction_set.discard("")
        sum_intersection = sum_intersection + len(
            ground_truth_set.intersection(prediction_set)
        )
        sum_prediction_and_ancestors = sum_prediction_and_ancestors + len(
            ground_truth_set
        )
    recall = sum_intersection / sum_prediction_and_ancestors
    return recall


def f1(y_true: np.ndarray, y_pred: np.ndarray):
    r"""
    Compute f1 score for hierarchical classification.

    :math:`\displaystyle{hF = \frac{2 \times hP \times hR}{hP + hR}}`,
    where :math:`hP` is the hierarchical precision and :math:`hR` is the hierarchical recall.

    Parameters
    ----------
    y_true : np.array of shape (n_samples, n_levels)
        Ground truth (correct) labels.
    y_pred : np.array of shape (n_samples, n_levels)
        Predicted labels, as returned by a classifier.
    Returns
    -------
    f1 : float
        Weighted average of the precision and recall
    """
    y_true, y_pred = _validate_input(y_true, y_pred)
    y_pred = y_pred.astype(int)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = 2 * prec * rec / (prec + rec + np.finfo(float).eps)
    return f1
