"""Prediction-to-class conversion and the multiclass metric helper."""

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def get_predicted(data):
    """ To convert probability values to predicted classes.
    """
    if len(data.shape) == 3:
        data = np.mean(data, axis=0)
    return np.argmax(data, -1)#np.where(data[:,1] >= threshold, 1, 0)


def multiclass_score(target, predicted_prob, score_type):
    """ To calculate several performance metrics for the multiclass task.
    """

    if len(predicted_prob.shape) == 1:
        predicted = np.where(predicted_prob >= 0.5, 1, 0)
        if score_type == 'f1':
            score = f1_score(target, predicted)
        elif score_type == 'recall':
            score = recall_score(target, predicted)
        elif score_type == 'precision':
            score = precision_score(target, predicted)
        elif score_type == 'auroc':
            score = roc_auc_score(target, predicted_prob)
        elif score_type == 'auprc':
            score = average_precision_score(target, predicted_prob)
        else:
            raise NotImplementedError
        return score
    else:
        class_list = np.unique(target)
        score_list = []
        predicted = get_predicted(predicted_prob)

        for pos_class in class_list:
            pos_class = int(pos_class)
            if pos_class == 0:
                continue

            tmp_target = (target == pos_class).astype(float)
            tmp_predicted = (predicted == pos_class).astype(float)
            tmp_predicted_prob = predicted_prob[:, pos_class]
            if score_type == 'f1':
                score_list.append(f1_score(tmp_target, tmp_predicted))
            elif score_type == 'recall':
                score_list.append(recall_score(tmp_target, tmp_predicted))
            elif score_type == 'precision':
                score_list.append(precision_score(tmp_target, tmp_predicted))
            elif score_type == 'auroc':
                score_list.append(roc_auc_score(tmp_target, tmp_predicted_prob))
            elif score_type == 'auprc':
                score_list.append(average_precision_score(tmp_target, tmp_predicted_prob))
            else:
                raise NotImplementedError()
        return np.mean(score_list)
