import os

import torch
import torch.nn as nn

import timm
import timm.loss as loss
from torch.utils.data import DataLoader
import torch.optim as optim
from lion_pytorch import Lion
from adabelief_pytorch import AdaBelief
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR, CosineAnnealingWarmRestarts
import wandb

from src.train import train
from src.val import val
from src.dataset import ClassificationDataset
from src.utils.metrics import log_metrics, log_wandb
from src.utils.callbacks import Callback
from src.utils.parse_args import parse_args

from src.augmentation.basic_aug import *

if __name__ == '__main__':
    args = parse_args()


    if args.augmentation == "a1":
        aug = augmentation
    elif args.augmentation == "a2":
        aug = augmentation2
    elif args.augmentation == "a3":
        aug = augmentation3
    else:
        aug = None
    train_dataset = ClassificationDataset('train', transformations=aug, dataset=args.dataset)
    test_dataset = ClassificationDataset('val', dataset=args.dataset)
    classes = train_dataset.class_num

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, prefetch_factor=args.pf_factor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, prefetch_factor=args.pf_factor)


    device = torch.device(args.cuda)
    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=classes)
    model.to(device)

    loss_dict = {"cross_entropy": nn.CrossEntropyLoss(),
                 "smoothing_cross_entropy":loss.LabelSmoothingCrossEntropy()}

    loss_fn = loss_dict[args.loss].to(device)

    epochs = args.epochs
    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    optimizer_dict = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "rmsprop": optim.RMSprop,
        "sgd": optim.SGD,
        "adabelief": AdaBelief,
        "lion": Lion
    }

    if args.optimizer in optimizer_dict:
        if args.optimizer == "sgd":
            optimizer = optimizer_dict[args.optimizer](model.parameters(), lr=args.lr, momentum=args.momentum,
                                                       weight_decay=args.w_decay)
        elif args.optimizer == "adabelief":
            optimizer = optimizer_dict[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.w_decay,
                                                       eps=1e-16, betas=(0.9, 0.999), weight_decouple=True,
                                                       rectify=False)
        else:
            optimizer = optimizer_dict[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    scheduler_dict = {
        "cosine": CosineAnnealingLR(optimizer, T_max=args.epochs),
        "polylr": PolynomialLR(optimizer, total_iters=args.epochs, power=args.power),
        "cosine_wr": CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=args.t0, T_mult=args.tm, eta_min=args.eta)
    }
    scheduler = scheduler_dict[args.scheduler]

    model_save_name =  (f"{args.model_save_name}${args.model}${args.loss}$"
                     f"{args.optimizer}${args.epochs}${args.augmentation}${args.scheduler}$"
                     f"{args.lr}${args.pretrained}${args.batch_size}${args.dataset}$")
    model_save_dir = args.models_dir
    os.makedirs(f"./{model_save_dir}", exist_ok=True)
    model_save_name = f"./{model_save_dir}/{model_save_name}"

    callbacks = Callback(patience=args.patience, model_save_name=model_save_name)

    wandb.init(
        project="CLASSIFICATION",
        config={
            "save_name": model_save_name,
            "model": args.model,
            "loss": args.loss,
            "optimizer": args.optimizer,
            "augmentation": args.augmentation,
            "scheduler": args.scheduler,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs
        },
        name=model_save_name
    )

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} || {epochs}")
        tl = train(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device, train_loader=train_loader)
        vl,  acc, prec, rec, f1 = val(model=model, loss_fn=loss_fn, device=device, val_loader=test_loader, classes=classes)
        log_metrics(tl, vl, acc, prec, rec, f1)
        log_wandb(tl, vl, acc, prec, rec, f1, epoch)
        scheduler.step()

        stop_training = callbacks.on_epoch_end(model, epoch, vl, f1, rec, prec)
        if stop_training:
            print("Early stopping triggered")
            break
    callbacks.on_train_end(model)
