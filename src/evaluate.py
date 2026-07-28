import torch 
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix,
)
from src.config import(
    NUM_CLASSES,
    CLASS_NAMES,
)
from src.dataset import create_dataloaders
from src.model import create_model

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, test_loader = create_dataloaders()
    model = create_model()
    model.load_state_dict(
        torch.load(
            "checkpoints/best_model.pth",
            map_location=device,
        )
    )
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())


    #metrics
    accuracy = accuracy_score(
        all_labels, 
        all_predictions,
    )

    print("\n====================")
    print("Evaluation Results")
    print("======================")

    print(f"\nAccracy: {accuracy * 100:.2f}%")
    print(f"\nClassification_report: \n")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=CLASS_NAMES,
        )
    )

    print("Confusion Matrix: \n")
    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )
    print(cm)

if __name__== "__main__":
    main()
