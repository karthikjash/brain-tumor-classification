import torch
import torch.nn as nn
import torch.optim as optim


from src.config import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

from src.dataset import create_dataloaders
from src.model import create_model
from src.engine import (
    train_one_epoch,
    validate_one_epoch,
)

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    train_loader, test_loader = create_dataloaders()
    model = create_model()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr = LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3,
    )

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc = validate_one_epoch(
            model, 
            test_loader,
            criterion,
            device,
        )

        scheduler.step(val_loss)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.2f}%"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% " 

        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                "checkpoints/best_model.pth",
            )
            print("Best model saved. ")

if __name__ == "__main__":
    main()