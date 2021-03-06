import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, confusion_matrix, \
    roc_curve, log_loss, balanced_accuracy_score, hamming_loss, hinge_loss, jaccard_score, explained_variance_score, \
    max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score


class Metrics:
    def __init__(self, y_test, y_pred, type):
        self.y_test = y_test
        self.y_pred = y_pred
        self.type = type

    def select_metrics(self, metrics_sel):
        if self.type == 'Classification':
            self.class_metrics = {
                'Accuracy': accuracy_score,
                'Recall': recall_score,
                'Precision': precision_score,
                'F1score': f1_score,
                'Confusion': confusion_matrix,
                'AUC': roc_auc_score,
                'ROC': roc_curve,
                'logloss': log_loss,
                'balancedaccuracy': balanced_accuracy_score,
                'hammingloss': hamming_loss,
                'hingeloss': hinge_loss,
                'jaccardscore': jaccard_score
            }
            self.metric = self.class_metrics[metrics_sel]
        else:
            self.regr_metrics = {
                'rmse': mean_squared_error,
                'jaccardscore': jaccard_score,
                'mae': mean_absolute_error,
                'mse': mean_squared_error,
                'medianabserror': median_absolute_error,
                'maxerror': max_error,
                'r2score': r2_score,
                'mean_squared_log_error': mean_squared_log_error,
                'explained_variance_score': explained_variance_score
            }
            self.metric = self.regr_metrics[metrics_sel]
            if metrics_sel == 'rmse':
                np.sqrt(self.metric)
        return self.metric

    def metrics_solve(self, **kwargs):
        return self.metric(self.y_test, self.y_pred, **kwargs)

