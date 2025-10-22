import torchmetrics
import wandb

class Metrics:
    def __init__(self, device, mode="val", classes=6):
        self.mode = mode
        self.accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=classes).to(device)
        self.precision_metric = torchmetrics.Precision(task="multiclass", num_classes=classes).to(device)
        self.recall_metric = torchmetrics.Recall(task="multiclass", num_classes=classes).to(device)
        self.f1_metric = torchmetrics.classification.F1Score(task="multiclass", num_classes=classes).to(device)
        if mode == "test":
            self.auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=classes).to(device)
    def batch_metrics(self, preds, labels):
        self.accuracy_metric.update(preds, labels)
        self.precision_metric.update(preds, labels)
        self.recall_metric.update(preds, labels)
        self.f1_metric.update(preds, labels)
        if self.mode == "test":
            self.auroc.update(preds, labels)

    def __reset(self):
        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()
        if self.mode == "test":
            self.auroc.reset()

    def epoch_metrics(self):
        acc = self.accuracy_metric.compute().item()
        prec = self.precision_metric.compute().item()
        rec = self.recall_metric.compute().item()
        f1 = self.f1_metric.compute().item()
        if self.mode == "test":
            auroc = self.auroc.compute().item()
        self.__reset()

        if self.mode == "test":
            return {"Accuracy":acc, "Precision":prec, "Recall":rec, "F1": f1, "AUROC": auroc}
        else:
            return acc, prec, rec, f1

def log_metrics(train_loss, val_loss, acc, prec, rec, f1):
    print(30*"-")
    print(f"Train Loss: {train_loss} | Validation Loss: {val_loss}")
    print(f"Accuracy: {acc} | Precision: {prec} | Recall: {rec} | F1: {f1}")
    print(30 * "-")

def log_wandb(train_loss, val_loss, acc, prec, rec, f1, epoch):
    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "epoch":epoch
    })



