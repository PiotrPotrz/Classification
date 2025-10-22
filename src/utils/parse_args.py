import argparse
import datetime


def parse_args():
    now = datetime.datetime.now()
    formatted_now = now.strftime("%Y_%m_%d %H_%M_%S")
    parser = argparse.ArgumentParser(
        description="Training script for multiclass classification."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument(
        "--model_save_name",
        type=str,
        default=formatted_now,
        help="Specify model save name",
    )
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    parser.add_argument("--model", type=str, default="resnet18", help="Specify model name")
    parser.add_argument("--loss", type=str, default="cross_entropy", help="Specify loss function")
    parser.add_argument(
        "--augmentation", type=str, default=None, help="Augmentation config"
    )
    parser.add_argument(
        "--cuda", type=str, default="cuda:0", help="Specify cuda device"
    )
    parser.add_argument(
        "--models_dir", type=str, default="saved_models", help="Specify model save directory"
    )
    parser.add_argument(
        "--pretrained", action='store_false', help="Use when you dont want to use pretrained model."
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine", help="Specify scheduler"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Specify patience"
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of workers used during dataset loading"
    )
    parser.add_argument(
        "--pf_factor", type=int, default=2, help="Number of pre fetch factor used during dataset loading"
    )
    parser.add_argument(
        "--pin_mem", action='store_false', help="Using pin memory?"
    )
    parser.add_argument(
        "--w_decay", type=float, default=0.0, help="Specify weight decay"
    )
    parser.add_argument(
        "--power", type=float, default=0.9, help="Specify the power while using POLY LR"
    )
    parser.add_argument(
        "--t0", type=int, default=80, help="Set t0 for warm restarts."
    )
    parser.add_argument(
        "--tm", type=int, default=2, help="Multiplication factor for warm restarts."
    )
    parser.add_argument(
        "--eta", type=float, default=1e-6, help="Eta for warm restarts."
    )
    parser.add_argument("--dataset", type=str, default="intel_image", help="Specify dataset name.")

    return parser.parse_args()

def parse_args_inference():
    parser = argparse.ArgumentParser(
        description="Testing script for multiclass classification."
    )
    parser.add_argument(
        "--cuda", type=str, default="cuda:0", help="Specify cuda device"
    )
    parser.add_argument(
        "--models_dir", type=str, default="saved_models", help="Specify model save directory"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of workers used during dataset loading"
    )
    parser.add_argument(
        "--pf_factor", type=int, default=8, help="Number of workers used during dataset loading"
    )
    parser.add_argument(
        "--pin_mem", action='store_false', help="Using pin memory?"
    )

    return parser.parse_args()
