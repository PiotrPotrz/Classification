import torch

class Callback:
    def __init__(self, patience, model_save_name):
        self.patience = patience
        self.model_save_name = model_save_name
        self.best_loss  = float("inf")
        self.best_f1 = 0
        self.best_precision = 0
        self.best_recall = 0
        self.continue_training = True
        self.no_improvement = 0

    def __one_op(self, model, metric_name):
        torch.save(model.state_dict(), f"{self.model_save_name}_best_{metric_name}.pth")
        self.no_improvement = 0

    def on_epoch_end(self, model, epoch, loss, f1, recall, precision):
        if epoch == 0:
            self.best_loss = loss
            self.best_f1 = f1
            self.best_recall = recall
            self.best_precision = precision
            torch.save(model.state_dict(), f"{self.model_save_name}_best_loss.pth")
            torch.save(model.state_dict(), f"{self.model_save_name}_best_f1.pth")
            torch.save(model.state_dict(), f"{self.model_save_name}_best_precision.pth")
            torch.save(model.state_dict(), f"{self.model_save_name}_best_recall.pth")
        else:
            self.no_improvement += 1
            if loss < self.best_loss:
                self.best_loss = loss
                self.__one_op(model, "loss")
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.__one_op(model, "f1")
            if recall > self.best_recall:
                self.best_recall = recall
                self.__one_op(model, "recall")
            if precision > self.best_precision:
                self.best_precision = precision
                self.__one_op(model, "precision")
            return self.no_improvement >= self.patience

    def on_train_end(self, model):
        torch.save(model.state_dict(), f"{self.model_save_name}_last.pth")