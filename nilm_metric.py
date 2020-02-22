import sys
import os

import numpy as np
from data_preprocessing import ActivationDetector

def get_nde(x, x_):
    return np.sum(np.square(x_ - x)) / np.sum(np.square(x))

def get_mae(x, x_):
    return np.sum(np.abs(x_ - x)) / len(x)

def get_sae(x, x_, sampling_interval=8):
    return np.abs(np.sum(x_) - np.sum(x)) / np.sum(x)

def get_dsae(x, x_, sampling_interval=8):
    day_samples = 60*60*24//sampling_interval
    x_days = np.reshape(x[:-(len(x) % day_samples)], (-1, day_samples))
    x__days = np.reshape(x_[:-(len(x) % day_samples)], (-1, day_samples))
    r = np.sum(x_days, 1)
    r_ = np.sum(x__days, 1)
    return np.sum(np.abs(r - r_)) / np.sum(x)

def get_classification_stats(results, appliance, sample_rate=8):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    rules_file = os.path.join(dir_path, '../../data_conversion/rules.json')
    activation_detector = ActivationDetector(appliance, rulesfile=rules_file, sample_rate=sample_rate)

    gt_activations = activation_detector.get_activations(results.ground_truth)
    rules_file = os.path.join(dir_path, '../../inferred_bam/rules.json')
    activation_detector = ActivationDetector(appliance, rulesfile=rules_file, sample_rate=sample_rate)
    pred_activations = activation_detector.get_activations(results.predictions)

    gt_activations['overlap'] = False
    pred_activations['overlap'] = False

    for gt_index, gt_row in gt_activations.iterrows():
        for pred_index, pred_row in pred_activations.iterrows():
            if (((pred_row.start >= gt_row.start)
                and (pred_row.start <= gt_row.end))
            or ((pred_row.end >= gt_row.start)
                and (pred_row.end <= gt_row.end))
            or ((pred_row.start <= gt_row.start)
                and (pred_row.end >= gt_row.end))):

                gt_activations.loc[gt_index, 'overlap'] = True
                pred_activations.loc[pred_index, 'overlap'] = True

    tp_gt = gt_activations.overlap.sum()
    tp_pred = pred_activations.overlap.sum()
    fp = (~pred_activations.overlap).sum()
    fn = (~gt_activations.overlap).sum()

    recall = gt_activations.overlap.mean() #tp / (tp + fn)
    precision = pred_activations.overlap.mean() #tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)

    return tp_gt, fp, fn, recall, precision, f1