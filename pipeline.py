"""Sanad's complete, deliberately small inference pipeline."""

import argparse
import json
import math
import os
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision import models, transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO


ROOT = Path(__file__).parent
WEIGHTS = ROOT / "weights"
FORM_FIELDS = {
    "maternal": ["weight_kg", "medications_taken"],
    "newborn": ["weight_g", "treatment"],
}
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def make_classifier():
    """Rebuild ResNet18 exactly as it was built during training."""
    checkpoint = torch.load(WEIGHTS / "form_classifier.pt", map_location="cpu", weights_only=True)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.fc.load_state_dict(checkpoint["final_layer"])
    model.eval()
    return model, checkpoint["classes"]


def load_models():
    """Load each model once so evaluation can reuse them for several images."""
    classifier, classes = make_classifier()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ocr_root = Path(os.environ.get("SANAD_OCR_MODELS", WEIGHTS))
    ocr_paths = {
        "latin": ocr_root / "trocr_latin",
        "numeric": ocr_root / "trocr_numbers",
    }
    for path in ocr_paths.values():
        if not (path / "model.safetensors").exists():
            raise FileNotFoundError(
                f"Missing fine-tuned OCR checkpoint: {path}. "
                "Set SANAD_OCR_MODELS to the folder containing trocr_latin and trocr_numbers."
            )

    processors = {name: TrOCRProcessor.from_pretrained(path, local_files_only=True) for name, path in ocr_paths.items()}
    ocr_models = {
        name: VisionEncoderDecoderModel.from_pretrained(path, local_files_only=True, use_safetensors=True).to(device).eval()
        for name, path in ocr_paths.items()
    }
    if device == "cuda":
        for model in ocr_models.values():
            model.half()
    return {
        "classifier": classifier.to(device),
        "classes": classes,
        "maternal": YOLO(str(WEIGHTS / "maternal_yolo.pt")),
        "newborn": YOLO(str(WEIGHTS / "newborn_yolo.pt")),
        "handwriting": YOLO(str(WEIGHTS / "handwriting_yolo.pt")),
        "ocr_processors": processors,
        "ocr_models": ocr_models,
        "device": device,
    }


def classify_form(image, loaded):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0).to(loaded["device"])
    with torch.no_grad():
        probabilities = loaded["classifier"](tensor).softmax(dim=1)[0]
    index = int(probabilities.argmax())
    return loaded["classes"][index], float(probabilities[index])


def detect(model, image, class_names):
    """Return plain dictionaries instead of exposing Ultralytics objects."""
    prediction = model.predict(image, imgsz=960, conf=0.25, verbose=False)[0]
    detections = []
    for box in prediction.boxes:
        class_id = int(box.cls.item())
        detections.append({
            "class_id": class_id,
            "name": class_names[class_id],
            "box": [round(float(value), 2) for value in box.xyxy[0]],
            "confidence": round(float(box.conf.item()), 4),
        })
    return detections


def center(box):
    """The center of [left, top, right, bottom]."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def match_labels_to_handwriting(labels, handwriting, image_size, max_distance_fraction=0.30):
    """Greedily pair each label with the nearest still-unused handwriting box."""
    width, height = image_size
    maximum_distance = math.sqrt(width**2 + height**2) * max_distance_fraction
    unused = set(range(len(handwriting)))
    matches = []

    # Handle the most certain printed labels first when detections overlap.
    for label in sorted(labels, key=lambda item: item["confidence"], reverse=True):
        label_x, label_y = center(label["box"])
        candidates = []
        for index in unused:
            hand_x, hand_y = center(handwriting[index]["box"])
            # Euclidean distance: d = sqrt((x1-x2)^2 + (y1-y2)^2).
            distance = math.sqrt((label_x - hand_x) ** 2 + (label_y - hand_y) ** 2)
            if distance <= maximum_distance:
                candidates.append((distance, index))

        if not candidates:
            matches.append({"label": label, "handwriting": None, "distance": None, "confidence": 0.0})
            continue
        distance, index = min(candidates)
        unused.remove(index)  # One answer cannot be assigned to two fields.
        matches.append({
            "label": label,
            "handwriting": handwriting[index],
            "distance": round(distance, 2),
            "confidence": round(1 - distance / maximum_distance, 4),
        })
    return matches


def crop_with_margin(image, box, margin_fraction=0.15):
    """Add a small margin so the OCR does not lose edge characters."""
    left, top, right, bottom = box
    margin_x = max(8, (right - left) * margin_fraction)
    margin_y = max(8, (bottom - top) * margin_fraction)
    expanded = (
        max(0, int(left - margin_x)), max(0, int(top - margin_y)),
        min(image.width, int(right + margin_x)), min(image.height, int(bottom + margin_y)),
    )
    return image.crop(expanded)


def read_handwriting(crop, loaded, route):
    """TrOCR uses a vision transformer encoder and text transformer decoder."""
    processor = loaded["ocr_processors"][route]
    model = loaded["ocr_models"][route]
    pixels = processor(images=crop, return_tensors="pt").pixel_values.to(
        loaded["device"], dtype=next(model.parameters()).dtype
    )
    with torch.no_grad():
        generated = model.generate(
            pixels, max_new_tokens=48, output_scores=True, return_dict_in_generate=True
        )
    text = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()

    # Each decoder step gives probabilities for its next character/token.
    # Their mean is a simple, understandable OCR confidence estimate.
    token_confidences = []
    for step, scores in enumerate(generated.scores):
        token = generated.sequences[0, -len(generated.scores) + step]
        token_confidences.append(float(scores[0].softmax(dim=0)[token]))
    confidence = sum(token_confidences) / len(token_confidences) if token_confidences else 0.0
    return text, round(confidence, 4)


def validate(field_name, raw_text):
    """Return a value, a confidence factor, and an explanation; always keep raw_text."""
    text = raw_text.strip().translate(ARABIC_DIGITS)
    compact = re.sub(r"\s+", "", text.lower()).replace(",", ".")

    if field_name in {"weight_kg", "weight_g"}:
        unit = "kg" if field_name == "weight_kg" else "g"
        number_text = re.sub(r"(?:kg|g)$", "", compact)
        correction = None

        if not re.fullmatch(r"\d+(?:\.\d+)?", number_text):
            return None, False, 0.0, "weight is not a number", None
        value = float(number_text)
        low, high = ((30, 200) if field_name == "weight_kg" else (500, 6000))

        # Example: OCR sees "51789" although the form says "5178g".
        # We report both strings, apply the plausible correction, and lower confidence.
        if field_name == "weight_g" and value > high and number_text.endswith("9"):
            possible_value = float(number_text[:-1])
            if low <= possible_value <= high:
                value = possible_value
                correction = "interpreted final 9 as the unit g"

        if not low <= value <= high:
            return value, False, 0.0, f"expected {low}-{high} {unit}", correction
        confidence_factor = 0.65 if correction else 1.0
        return value, True, confidence_factor, f"valid {unit} weight", correction

    # For this demo, a free-text answer is valid when it contains two letters.
    letters = sum(character.isalpha() for character in text)
    if letters < 2:
        return text or None, False, 0.0, "text is empty or too short", None
    return text, True, 1.0, "non-empty text", None


def annotate(image, labels, handwriting, matches, destination):
    drawing = ImageDraw.Draw(image)
    for item in labels:
        drawing.rectangle(item["box"], outline="red", width=4)
        drawing.text((item["box"][0], item["box"][1] - 16), item["name"], fill="red")
    for item in handwriting:
        drawing.rectangle(item["box"], outline="blue", width=3)
    for match in matches:
        if match["handwriting"]:
            drawing.line([center(match["label"]["box"]), center(match["handwriting"]["box"])], fill="green", width=3)
    image.save(destination)


def run_pipeline(image_path, loaded=None, save_outputs=True, verbose=True):
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    loaded = loaded or load_models()

    form_type, form_confidence = classify_form(image, loaded)
    labels = detect(loaded[form_type], image, FORM_FIELDS[form_type])
    handwriting = detect(loaded["handwriting"], image, ["handwriting"])
    matches = match_labels_to_handwriting(labels, handwriting, image.size)

    fields = {}
    for field_name in FORM_FIELDS[form_type]:
        match = next((item for item in matches if item["label"]["name"] == field_name), None)
        if not match or not match["handwriting"]:
            fields[field_name] = {
                "raw_text": None, "value": None, "confidence": 0.0,
                "valid": False, "needs_review": True, "message": "no handwriting matched",
            }
            continue

        crop = crop_with_margin(image, match["handwriting"]["box"])
        ocr_route = "numeric" if field_name in {"weight_kg", "weight_g"} else "latin"
        raw_text, ocr_confidence = read_handwriting(crop, loaded, ocr_route)
        value, valid, validation_confidence, message, correction = validate(field_name, raw_text)
        stage_confidences = {
            "form_classifier": form_confidence,
            "label_detector": match["label"]["confidence"],
            "handwriting_detector": match["handwriting"]["confidence"],
            "matching": match["confidence"],
            "ocr": ocr_confidence,
            "validation": validation_confidence,
        }
        overall_confidence = min(stage_confidences.values())
        fields[field_name] = {
            "raw_text": raw_text,
            "value": value,
            "correction": correction,
            "confidence": round(overall_confidence, 4),
            "stage_confidences": {key: round(value, 4) for key, value in stage_confidences.items()},
            "valid": valid,
            "needs_review": (not valid) or correction is not None or overall_confidence < 0.70,
            "message": message,
        }

    result = {
        "image": str(image_path),
        "form_type": form_type,
        "form_confidence": round(form_confidence, 4),
        "label_detections": labels,
        "handwriting_detections": handwriting,
        "matches": matches,
        "fields": fields,
    }

    if verbose:
        print(f"Predicted form: {form_type} ({form_confidence:.3f})")
        print("Field labels:", json.dumps(labels, indent=2))
        print("Handwriting regions:", json.dumps(handwriting, indent=2))
        print("Matches:", json.dumps(matches, indent=2))
        for name, field in fields.items():
            print(f"{name}: OCR={field['raw_text']!r}, value={field['value']!r}, "
                  f"valid={field['valid']}, confidence={field['confidence']:.3f}, "
                  f"needs_review={field['needs_review']}")

    if save_outputs:
        json_path = image_path.with_name(image_path.stem + "_result.json")
        annotated_path = image_path.with_name(image_path.stem + "_annotated.jpg")
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        annotate(image.copy(), labels, handwriting, matches, annotated_path)
        print(f"Saved {json_path} and {annotated_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to a maternal or newborn form image")
    args = parser.parse_args()
    run_pipeline(args.image)


if __name__ == "__main__":
    main()
