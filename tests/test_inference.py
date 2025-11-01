def test_inference_runs():
    import torch
    from src.utils.metrics import Metrics
    import timm

    model = timm.create_model('resnet18', num_classes=10)
    X = torch.randn(1, 3, 64, 64)
    y = torch.randint(0, 10, (1,))

    outputs = model(X)

    metrics = Metrics(device="cpu", mode="test", classes=10)
    metrics.batch_metrics(outputs, y)
    res = metrics.epoch_metrics()

    assert "Accuracy" in res
    assert "Precision" in res
    assert "Recall" in res
    assert "AUROC" in res