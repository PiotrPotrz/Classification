import torch
from tqdm import tqdm

def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc="   Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        loop.set_postfix(loss=loss.item())
    return train_loss / len(train_loader)
