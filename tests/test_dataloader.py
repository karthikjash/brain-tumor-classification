from src.dataset import create_dataloaders

train_loader, test_loader = create_dataloaders()

images, labels = next(iter(train_loader))

print("Images Shape :", images.shape)
print("Labels Shape :", labels.shape)

print("Labels:", labels[:10])