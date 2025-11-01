import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def test_train_pass():
    from src.train import train
    from src.val import val

    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 2)
    )

    X = torch.randn(8, 3, 64, 64)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(X, y), batch_size=2)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    tl = train(model=model, loss_fn=loss_fn, optimizer=optimizer,
               device=device, train_loader=loader)
    vl, acc, prec, rec, f1 = val(model=model, loss_fn=loss_fn, device=device, val_loader=loader, classes=2)

    assert tl is None or isinstance(tl, (float, dict))
    assert vl is None or isinstance(vl, (float, dict))
    assert acc is None or isinstance(acc, (float, dict))
    assert prec is None or isinstance(prec, (float, dict))
    assert rec is None or isinstance(rec, (float, dict))
    assert f1 is None or isinstance(f1, (float, dict))
