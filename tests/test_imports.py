import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
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

    assert callable(train)
    assert callable(val)
    assert callable(parse_args)
    assert callable(log_metrics)
    assert callable(log_wandb)
    assert callable(Callback)
    assert callable(ClassificationDataset)

