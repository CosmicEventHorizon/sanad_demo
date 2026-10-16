"""Minimal held-out evaluation. Run: python evaluate.py"""

import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import datasets, transforms
from ultralytics import YOLO

from pipeline import WEIGHTS, classify_form, load_models, run_pipeline, validate


ROOT = Path(__file__).parent


def levenshtein(reference, prediction):
    """Minimum insertions, deletions and substitutions needed to change one string."""
    previous = list(range(len(prediction) + 1))
    for row, reference_character in enumerate(reference, start=1):
        current = [row]
        for column, prediction_character in enumerate(prediction, start=1):
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            substitution = previous[column - 1] + (reference_character != prediction_character)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def intersection_over_union(first, second):
    """IoU = overlapping area / combined area; 1 is perfect and 0 is no overlap."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection) if intersection else 0.0


def classifier_accuracy(loaded):
    test_data = datasets.ImageFolder(
        ROOT / "data/classifier/test",
        transform=transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    )
    correct = 0
    for image_path, correct_index in test_data.samples:
        predicted, _ = classify_form(Image.open(image_path).convert("RGB"), loaded)
        correct += predicted == test_data.classes[correct_index]
    return correct / len(test_data)


def resolved_yolo_yaml(name):
    source = ROOT / "data/yolo" / name
    description = yaml.safe_load((source / "dataset.yaml").read_text())
    description["path"] = str(source.resolve())
    destination = WEIGHTS / f"{name}_evaluation.yaml"
    destination.write_text(yaml.safe_dump(description, sort_keys=False))
    return destination


def main():
    if not all((WEIGHTS / name).exists() for name in [
        "form_classifier.pt", "maternal_yolo.pt", "newborn_yolo.pt", "handwriting_yolo.pt"
    ]):
        raise FileNotFoundError("Train the models first: python train_classifier.py && python train_yolo.py")

    loaded = load_models()
    print(f"Form classification accuracy: {classifier_accuracy(loaded):.3f}")

    # Ultralytics computes detection precision, recall and mAP at IoU thresholds.
    for name in ["maternal", "newborn", "handwriting"]:
        metrics = YOLO(str(WEIGHTS / f"{name}_yolo.pt")).val(
            data=str(resolved_yolo_yaml(name)), split="test", imgsz=960, device=0 if torch.cuda.is_available() else "cpu", verbose=False
        )
        print(f"{name} YOLO: precision={metrics.box.mp:.3f}, recall={metrics.box.mr:.3f}, "
              f"mAP@IoU50={metrics.box.map50:.3f}")

    truth = json.loads((ROOT / "data/ground_truth.json").read_text())
    correct_matches = total_fields = total_characters = total_character_errors = exact = 0
    hand_ious = []
    for example in truth:
        result = run_pipeline(ROOT / "data" / example["image"], loaded, save_outputs=False, verbose=False)
        image = Image.open(ROOT / "data" / example["image"])
        for field_name, expected in example["fields"].items():
            total_fields += 1
            match = next((item for item in result["matches"] if item["label"]["name"] == field_name), None)
            if match and match["handwriting"]:
                predicted_box = match["handwriting"]["box"]
                expected_box = [expected["handwriting_box"][0] * image.width, expected["handwriting_box"][1] * image.height,
                                expected["handwriting_box"][2] * image.width, expected["handwriting_box"][3] * image.height]
                iou = intersection_over_union(predicted_box, expected_box)
                hand_ious.append(iou)
                correct_matches += iou >= 0.5

            predicted_text = result["fields"].get(field_name, {}).get("raw_text") or ""
            reference_text = str(expected["text"])
            total_character_errors += levenshtein(reference_text, predicted_text)
            total_characters += len(reference_text)

            expected_value = validate(field_name, reference_text)[0]
            predicted_value = result["fields"].get(field_name, {}).get("value")
            exact += str(predicted_value).casefold() == str(expected_value).casefold()

    print(f"Field-to-handwriting matching accuracy: {correct_matches / total_fields:.3f}")
    print(f"Mean matched-box IoU: {sum(hand_ious) / len(hand_ious) if hand_ious else 0:.3f}")
    print(f"Character Error Rate (CER): {total_character_errors / total_characters:.3f}")
    print(f"Final field exact-match accuracy: {exact / total_fields:.3f}")


if __name__ == "__main__":
    main()
