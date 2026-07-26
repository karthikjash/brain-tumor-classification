from src.config import TRAIN_DIR
from src.dataset import BrainTumorDataset
from src.preprocess import get_train_transforms

dataset = BrainTumorDataset(
    root_dir=TRAIN_DIR,
    transform=get_train_transforms()
)

print("Dataset Size:", len(dataset))

image, label = dataset[0]

print("Image Shape:", image.shape)
print("Label:", label)