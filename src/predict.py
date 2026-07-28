import argparse
import torch
from PIL import Image
from src.config import CLASS_NAMES
from src.model import create_model
from src.preprocess import get_test_transforms

def predict(image_path):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"

    )

    model = create_model()
    model.load_state_dict(
        torch.load(
            "checkpoints/best_model.pth",
            map_location=device,
        )
    )
    model.to(device)
    model.eval()

    transform = get_test_transforms()

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    #prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim = 1)
        confidence, prediction = torch.max(probabilities, dim=1)

    predicted_class = CLASS_NAMES[prediction.item()]

    print("\n==============================")
    print("Prediction Result")
    print("==============================")

    print(f"Image       : {image_path}")
    print(f"Prediction  : {predicted_class}")
    print(f"Confidence  : {confidence.item()*100:.2f}%")
    print("\nClass Probabilities:")

    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name:<12}: {probabilities[0][i].item()*100:.2f}%")


def main():

    parser = argparse.ArgumentParser(
        description="Brain Tumor MRI Prediction"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to MRI image"
    )

    args = parser.parse_args()
    predict(args.image_path)


if __name__ == "__main__":
    main()