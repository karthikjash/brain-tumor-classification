from pathlib import Path
from PIL import Image
from src.preprocess import get_train_transforms

image_path = next(Path("data/raw/Training/glioma").glob("*"))

image = Image.open(image_path)

transform = get_train_transforms()
tensor = transform(image)

print("Tensor Shape:", tensor.shape)
print("Tensor Dtype:", tensor.dtype)


