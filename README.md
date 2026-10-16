# Sanad demo

A small educational proof of concept that reads two fields from two healthcare forms.

| Form | Field 1 | Field 2 |
|---|---|---|
| Maternal | `weight_kg` | `medications_taken` |
| Newborn | `weight_g` | `treatment` |

The pipeline is deliberately direct:

```text
image -> ResNet18 form classifier -> form-specific label YOLO
      -> general handwriting YOLO -> nearest-box matching
      -> handwritten crop -> TrOCR -> validation -> JSON
```

ResNet18's frozen backbone converts an image into useful visual features. Its trainable final layer produces two scores, and softmax turns them into probabilities. Cross-entropy penalizes a low probability for the correct form.

YOLO predicts a class, box and confidence in one pass. There are two two-class label detectors and one one-class handwriting detector. A label and answer are paired by center distance:

`d = sqrt((x_label - x_hand)^2 + (y_label - y_hand)^2)`

The nearest unused handwriting box inside 30% of the image diagonal is selected. TrOCR then uses a vision-transformer encoder and text-transformer decoder to read only that crop.

Validation keeps the raw OCR text. For example, an impossible newborn weight `51789` may be reported as value `5178 g` because the final `9` resembles `g`, but confidence is reduced and `needs_review` is true. A field's final confidence is the minimum confidence of every stage, so one weak stage cannot be hidden by the others.

## Install and train

```bash
cd sanad_demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_classifier.py          # CUDA is required
python train_yolo.py                # trains all 3 YOLOs on CUDA
```

The copied dataset contains 320 training, 80 validation and 40 test pages. It is synthetic and intended only for this technical demonstration, not clinical use.

## Run

```bash
python pipeline.py --image data/evaluation/images/05_000017.jpg
python evaluate.py
```

The pipeline prints every intermediate result and saves `<name>_result.json` plus `<name>_annotated.jpg` beside the input. `evaluate.py` reports classifier accuracy, YOLO precision/recall/mAP, matching accuracy, mean IoU, character error rate (CER), and final exact-match accuracy on the included held-out annotations.

## Current held-out result

| Metric | Result |
|---|---:|
| Form classification accuracy (40 pages) | 1.000 |
| Maternal label YOLO precision / recall / mAP50 | 0.996 / 1.000 / 0.995 |
| Newborn label YOLO precision / recall / mAP50 | 0.990 / 0.999 / 0.995 |
| Handwriting YOLO precision / recall / mAP50 | 0.990 / 0.996 / 0.995 |
| Matching accuracy (8 fields) | 0.750 |
| Mean matched-box IoU | 0.695 |
| TrOCR character error rate | 1.340 |
| Final exact-match accuracy | 0.000 |

The last two rows are intentionally reported rather than hidden: pretrained TrOCR was not fine-tuned, as required, and is the clear bottleneck on these synthetic handwriting styles. CER can exceed 1 because the OCR may insert more wrong characters than the reference contains.
