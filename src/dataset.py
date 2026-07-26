from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from src.config import CLASS_NAMES

from torch.utils.data import DataLoader
from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
)
from src.preprocess import (
    get_train_transforms,
    get_test_transforms,
) 

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.class_to_idx = {
            class_name : idx 
            for idx, class_name in enumerate(CLASS_NAMES)

        }

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        for class_name in CLASS_NAMES:
            class_dir = self.root_dir / class_name

            for image_path in class_dir.iterdir():
                if image_path.is_file():
                    self.samples.append(
                        (
                            image_path, self.class_to_idx[class_name]
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders():

    train_dataset = BrainTumorDataset(
        root_dir=TRAIN_DIR,
        transform=get_train_transforms(),

    )
    test_dataset = BrainTumorDataset(
        root_dir=TEST_DIR,
        transform=get_test_transforms(),
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return train_loader, test_loader