import torch
import skimage
import numpy as np

from torch.utils.data import Dataset, DataLoader
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import os
import yaml

class ClassificationDataset(Dataset):
    def __init__(self, mode, dataset="intel_image", normalization="minmax", transformations=None,
                 random_state=42, val_size=0.1):
        self.dataset_name = dataset
        self.mode = mode
        self.transformations = transformations
        try:
            with open(f"./data/configs/{self.dataset_name.lower()}.yaml") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            print("Wrong dataset or config file! ", e)
    
        self.train_dataset_path = cfg["paths"]["train_dataset_path"]
        self.test_dataset_path = cfg["paths"]['test_dataset_path']
        self.val_dataset_path = cfg["paths"]['val_dataset_path']

        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(self.train_dataset_path))
        if (self.mode in ["train", "val"]) and self.val_dataset_path is None:
            self.__get_images_and_labels(self.train_dataset_path)
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                self.image_paths, self.labels, test_size=val_size, stratify=self.labels, random_state=random_state
            )
            if self.mode == "train":
                self.image_paths = train_paths
                self.labels = train_labels
            else:
                self.image_paths = val_paths
                self.labels = val_labels
        elif self.mode == "train":
            self.__get_images_and_labels(self.train_dataset_path)
        elif self.mode == "val":
            self.__get_images_and_labels(self.val_dataset_path)
        elif self.mode == "test":
            self.__get_images_and_labels(self.test_dataset_path)
        else:
            raise Exception("Wrong mode")

        self.labels = [cfg['class_to_idx'][cls] for cls in self.labels]
        self.class_num = cfg['class_num']
        self.image_size = cfg['img_size']

        if normalization == "minmax":
            self.base_transformations = [
                A.Normalize(normalization='min_max'),
                A.Resize(self.image_size, self.image_size, p=1.0),
                ToTensorV2(),
            ]
        else:
            raise NotImplementedError

        if self.transformations is not None:
            self.transform = A.Compose(self.transformations + self.base_transformations)
        else:
            self.transform = A.Compose(self.base_transformations)

    def __get_images_and_labels(self, dataset):
        for cls in self.classes:
            cls_dir = os.path.join(dataset, cls)
            for path in glob.glob(os.path.join(cls_dir, "*.jpg")):
                self.image_paths.append(path)
                self.labels.append(cls)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = skimage.io.imread(image_path)
        image = resize(image, (self.image_size, self.image_size),
                                         anti_aliasing=True, preserve_range=True).astype(np.uint8)

        image = self.transform(image=image)["image"]
        label = torch.tensor(label, dtype=torch.long)

        return image, label

if __name__ == "__main__":
    train_dataset = ClassificationDataset("train", normalization="minmax", dataset="INTEL_IMG")
    val_dataset = ClassificationDataset("val", normalization="minmax", dataset="INTEL_IMG")
    test_dataset = ClassificationDataset("test", normalization="minmax", dataset="INTEL_IMG")

    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print("Train loader size:", len(train_loader))
    print("Val loader size:", len(val_loader))
    print("Test loader size:", len(test_loader))
    for image, label in train_loader:
        print(f"image shape: {image.shape} || label shape: {label.shape}")
        print(f"image min: {image.min()} || max: {image.max()}")
        print("Label:", label)
        break

    for image, label in val_loader:
        print(f"image shape: {image.shape} || label shape: {label.shape}")
        print(f"image min: {image.min()} || max: {image.max()}")
        print("Label:", label)
        break

    for image, label in test_loader:
        print(f"image shape: {image.shape} || label shape: {label.shape}")
        print(f"image min: {image.min()} || max: {image.max()}")
        print("Label:", label)
        break

