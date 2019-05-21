import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score

import scikitplot as skplt

class Project:
    def extract_final_losses(self, history):
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        idx_min_val_loss = np.argmin(val_loss)
        return {'train_loss': train_loss[idx_min_val_loss], 'val_loss': val_loss[idx_min_val_loss]}

    def plot_training_error_curves(self, history):
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        _, ax = plt.subplots()
        ax.plot(train_loss, label='Train')
        ax.plot(val_loss, label='Validation')
        ax.set(title='Training and Validation Error Curves', xlabel='Epochs', ylabel='Loss (MSE)')
        ax.legend()
        plt.show()

    def compute_performance_metrics(self, y, y_pred_class, y_pred_scores=None):
        accuracy = accuracy_score(y, y_pred_class)
        recall = recall_score(y, y_pred_class)
        precision = precision_score(y, y_pred_class)
        f1 = f1_score(y, y_pred_class)
        performance_metrics = (accuracy, recall, precision, f1)
        if y_pred_scores is not None:
            skplt.metrics.plot_ks_statistic(y, y_pred_scores)
            plt.show()
            y_pred_scores = y_pred_scores[:, 1]
            auroc = roc_auc_score(y, y_pred_scores)
            aupr = average_precision_score(y, y_pred_scores)
            performance_metrics = performance_metrics + (auroc, aupr)
        return performance_metrics

    def print_metrics_summary(self, accuracy, recall, precision, f1, auroc=None, aupr=None):
        print()
        print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))
        print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))
        print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision))
        print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))
        if auroc is not None:
            print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))
        if aupr is not None:
            print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))

    def display_metrics(self, classifier, test_X, test_y, history=None, display_losses=False, mode=0):
        y_pred_class = classifier.predict(test_X)
        y_pred_scores = classifier.predict_proba(test_X)

        if mode == 0:
            y_pred_scores = classifier.predict(test_X)
            y_pred_class = classifier.predict_classes(test_X, verbose=0)
            y_pred_scores_0 = 1 - y_pred_scores
            y_pred_scores = np.concatenate([y_pred_scores_0, y_pred_scores], axis=1)

        print("Matriz de confusao no conjunto de teste:")
        print(confusion_matrix(test_y, y_pred_class))

        if display_losses:
            losses = self.extract_final_losses(history)
            print()
            print("{metric:<18}{value:.4f}".format(metric="Train Loss:", value=losses['train_loss']))
            print("{metric:<18}{value:.4f}".format(metric="Validation Loss:", value=losses['val_loss']))

        print('\nPerformance no conjunto de teste:')
        accuracy, recall, precision, f1, auroc, aupr = self.compute_performance_metrics(test_y, y_pred_class, y_pred_scores)
        self.print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
