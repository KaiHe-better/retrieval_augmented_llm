import torch
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score

class My_Metrics():
    def metrics_task_res(self, pred, label):

        acc = accuracy_score(label, pred)
        precision = precision_score(label, pred, average="macro")
        recall = recall_score(label, pred, average="macro")
        f1 = f1_score(label, pred, average="macro")

        acc = round(acc*100, 2)
        precision = round(precision*100, 2)
        recall = round(recall*100, 2)
        f1 = round(f1*100, 2)

        return acc, precision, recall, f1