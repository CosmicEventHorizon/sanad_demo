"""Train the two-class form classifier. Run: python train_classifier.py"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


ROOT = Path(__file__).parent
DATA = ROOT / "data/classifier"
WEIGHTS = ROOT / "weights"


def make_model():
    # A classifier turns pixels into features, then its last layer converts those
    # features into two scores (logits): one for maternal and one for newborn.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # The pretrained convolutional backbone already knows useful visual features
    # such as edges, shapes and textures. Freezing means its numbers do not change.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Only this new final layer is learned for our two form types.
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training this demo.")
    device = torch.device("cuda")

    # ResNet expects 224x224 tensors normalized like its ImageNet training data.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_data = datasets.ImageFolder(DATA / "train", transform=train_transform)
    validation_data = datasets.ImageFolder(DATA / "val", transform=validation_transform)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation_data, args.batch_size, num_workers=2)

    model = make_model().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    best_accuracy = 0.0
    best_final_layer = None

    for epoch in range(args.epochs):
        # Keep the frozen backbone in evaluation mode too. Otherwise BatchNorm's
        # running averages would change even though its learned weights are frozen.
        model.eval()
        model.fc.train()
        for images, correct_classes in train_loader:
            images, correct_classes = images.to(device), correct_classes.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, correct_classes)
            loss.backward()       # Compute how each final-layer weight affected the error.
            optimizer.step()      # Move those weights in the error-reducing direction.

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, correct_classes in validation_loader:
                logits = model(images.to(device))
                predictions = logits.argmax(dim=1).cpu()
                correct += (predictions == correct_classes).sum().item()
                total += len(correct_classes)
        accuracy = correct / total
        print(f"epoch {epoch + 1}/{args.epochs}: validation accuracy = {accuracy:.3f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_final_layer = {name: value.detach().cpu().clone() for name, value in model.fc.state_dict().items()}

    WEIGHTS.mkdir(exist_ok=True)
    # Only the final layer changed, so saving the frozen backbone again would waste
    # about 45 MB. Inference reloads the same public pretrained backbone.
    torch.save({"final_layer": best_final_layer, "classes": train_data.classes}, WEIGHTS / "form_classifier.pt")
    print("saved weights/form_classifier.pt")


if __name__ == "__main__":
    main()
