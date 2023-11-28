import torch
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score

class My_Metrics():
    def metrics_task_res(self, pred, label):

        acc = accuracy_score(label, pred)
        precision = precision_score(label, pred, average="macro")
        recall = recall_score(label, pred, average="macro")
        f1 = f1_score(label, pred, average="macro")

        return acc, precision, recall, f1
    