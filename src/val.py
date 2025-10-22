from tqdm import tqdm
import torch
from src.utils.metrics import Metrics

def val(val_loader, model, loss_fn, device, classes=6):
    metrics = Metrics(device=device, classes=classes)
    model.eval()
    val_loss = 0
    loop = tqdm(val_loader, desc="   Validation", leave=False)

    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            val_loss += loss.item()

            metrics.batch_metrics(outputs, labels)

        acc, prec, rec, f1 = metrics.epoch_metrics()
    return val_loss / len(val_loader), acc, prec, rec, f1