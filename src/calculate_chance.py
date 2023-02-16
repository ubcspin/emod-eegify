import numpy as np
import pandas as pd


def precision(y_true: np.ndarray, y_pred: np.ndarray):
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        ground_truth_set = set(ground_truth)
        prediction_set = set(prediction)
        sum_intersection = sum_intersection + len(
            ground_truth_set.intersection(prediction_set)
        )
        sum_prediction_and_ancestors = sum_prediction_and_ancestors + len(
            prediction_set
        )
    precision = sum_intersection / sum_prediction_and_ancestors
    return precision


def recall(y_true: np.ndarray, y_pred: np.ndarray):
    sum_intersection = 0
    sum_prediction_and_ancestors = 0
    for ground_truth, prediction in zip(y_true, y_pred):
        ground_truth_set = set(ground_truth)
        prediction_set = set(prediction)
        sum_intersection = sum_intersection + len(
            ground_truth_set.intersection(prediction_set)
        )
        sum_prediction_and_ancestors = sum_prediction_and_ancestors + len(
            ground_truth_set
        )
    recall = sum_intersection / sum_prediction_and_ancestors
    return recall


def f1(y_true: np.ndarray, y_pred: np.ndarray):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = 2 * prec * rec / (prec + rec)
    return f1



def row(pnum, window, scores):

    mean_f1, mean_prec, mean_recall = scores.mean(axis=0)
    std_f1, std_prec, std_recall = scores.std(axis=0)
    d = {
        'pnum': pnum,
        'window': window,

        
        'chance_f1': mean_f1,
        'std_chance_f1': std_f1,

        'chance_precision': mean_prec,
        'std_chance_precision': std_prec,


        'chance_recall': mean_recall,
        'std_chance_recall': std_recall
        }
    return pd.Series({**d})



if __name__ == "__main__":
    metric = lambda yt,yp: [f1(yt,yp), precision(yt,yp), recall(yt,yp)]
    df_dataset = pd.read_csv('cnn_dataset.csv')

    columns = ['pnum', 'window', 'chance_f1', 'std_chance_f1','chance_precision', 'std_chance_precision', 'chance_recall', 'std_chance_recall']
    rows = []
    for pnum in SUBJECT_IDS:
        for window in ['500', '5000']:
            n = 1000

            labelset = df_dataset[(df_dataset['pnum'] == pnum) & (df_dataset['window'] == window)][['cw', 'dir']]
            labels = df_dataset[(df_dataset['pnum'] == pnum) & (df_dataset['window'] == window)][['cw', 'dir']].values

            cw_unique = labelset.cw.unique() # number of unique words
            dir_unique = ['0','1','2'] # number of unique directions

            metrics = np.zeros((n,3))

            cw_pred = np.random.choice(cw_unique, size=(labels.shape[0], n))
            dir_pred = np.random.choice(dir_unique, size=(labels.shape[0], n))

            for i in range(n):
                pred = np.vstack([cw_pred[:,i],dir_pred[:,i]]).T
                metrics[i] = metric(labels, pred)

            rows.append((row(pnum, window, metrics)))

    df_chances = pd.concat(rows, axis=1).T.sort_values(['pnum'], ascending=False)

    df_chances.to_csv('chance_values.csv', index=False)