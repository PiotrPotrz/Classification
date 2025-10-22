import torch
import timm
from tqdm import tqdm
import glob
from torch.utils.data import DataLoader
import pandas as pd
import datetime
import os

import warnings
warnings.filterwarnings("ignore", message="UnsupportedFieldAttributeWarning")

from src.utils.parse_args import parse_args_inference
from src.dataset import ClassificationDataset
from src.utils.metrics import Metrics
from src.utils.utils import  add_result

if __name__=='__main__':
    args = parse_args_inference()
    device = torch.device(args.cuda)

    formatted_now = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")


    columns = ["SaveName", "Model",
               "Loss", "Optimizer", "Epochs",
               "Augmentation", "Scheduler", "lr",
               "Pretrained", "Batch size",
               "Dataset", "Accuracy", "Precision", "Recall", "F1", "AUROC",
               "type", "path"]

    results_df = pd.DataFrame(columns=columns)

    number_of_models = len(glob.glob(f'./{args.models_dir}/*.pth'))
    print(f"Found {number_of_models} models to perform inference on in {args.models_dir}")
    for i, path in enumerate(sorted(glob.glob(f'{args.models_dir}/*.pth'))):
        name_split = path.split('$')
        print(f"[{i+1}/{number_of_models}] Evaluating model: {name_split[1]} on dataset {name_split[10]} - checkpoint: {name_split[-1]}")
        model = timm.create_model(name_split[1], num_classes=6).to(device)
        test_dataset = ClassificationDataset('test', dataset=name_split[10])

        model.load_state_dict(torch.load(path))

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, prefetch_factor=args.pf_factor)

        metrics = Metrics(device=device, mode="test")
        model.eval()
        loop = tqdm(test_loader, desc="   Testing", leave=False)

        with torch.no_grad():
            for images, labels in loop:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                metrics.batch_metrics(outputs, labels)

            results = metrics.epoch_metrics()
        results_df = add_result(results_df, path, results)
        print(f"Accuracy: {results["Accuracy"]} | Precision {results["Precision"]} |  Recall {results["Recall"]} | F1 {results["F1"]} | AUROC {results["AUROC"]}")
    os.makedirs('./scores', exist_ok=True)
    results_df.to_csv(f"./scores/{args.models_dir}_{formatted_now}_results_inference.csv", index=False)