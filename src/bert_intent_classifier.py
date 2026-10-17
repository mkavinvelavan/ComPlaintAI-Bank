"""BERT intent classification using local files or a pinned deployment artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.model_artifacts import resolve_bert_model_dir


DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[1] / "models" / "bert_banking77_model"
)
DEFAULT_MAX_LENGTH = 128
EXPECTED_NUM_LABELS = 77


class BertIntentClassifier:
    """Classify complaint text using a locally saved fine-tuned BERT model."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = resolve_bert_model_dir(model_dir)

        required_files = ("config.json", "model.safetensors", "label_mapping.json")
        missing_files = [name for name in required_files if not (self.model_dir / name).is_file()]
        if missing_files:
            missing = ", ".join(missing_files)
            raise FileNotFoundError(
                f"Missing required model file(s) in {self.model_dir}: {missing}"
            )

        self.label_mapping = self._load_label_mapping(
            self.model_dir / "label_mapping.json"
        )
        self.labels = [self.label_mapping[index] for index in range(EXPECTED_NUM_LABELS)]
        self.max_length = self._load_max_length(self.model_dir / "model_metadata.json")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir), local_files_only=True, use_safetensors=True
            )
        except Exception:
            raise RuntimeError(
                "Could not load the configured fine-tuned BERT tokenizer/model."
            ) from None

        model_num_labels = getattr(self.model.config, "num_labels", None)
        if model_num_labels != EXPECTED_NUM_LABELS:
            raise ValueError(
                f"Model must have {EXPECTED_NUM_LABELS} output classes, but its "
                f"configuration reports {model_num_labels!r}."
            )
        if len(self.labels) != model_num_labels:
            raise ValueError(
                f"Label mapping contains {len(self.labels)} labels but the model "
                f"has {model_num_labels} output classes."
            )

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_label_mapping(mapping_path: Path) -> dict[int, str]:
        try:
            with mapping_path.open("r", encoding="utf-8") as mapping_file:
                data: Any = json.load(mapping_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read label mapping at {mapping_path}: {exc}") from exc

        raw_mapping = data.get("id_to_label") if isinstance(data, dict) else None
        if not isinstance(raw_mapping, dict):
            raise ValueError(
                f"Label mapping at {mapping_path} must contain an 'id_to_label' object."
            )

        try:
            mapping = {int(index): label for index, label in raw_mapping.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Label mapping at {mapping_path} contains a non-integer class ID."
            ) from exc

        expected_ids = set(range(EXPECTED_NUM_LABELS))
        if set(mapping) != expected_ids:
            raise ValueError(
                f"Label mapping must contain exactly class IDs 0 through "
                f"{EXPECTED_NUM_LABELS - 1}; found {len(mapping)} distinct IDs."
            )
        if any(not isinstance(label, str) or not label.strip() for label in mapping.values()):
            raise ValueError(f"Label mapping at {mapping_path} contains an empty or invalid label.")
        if len(set(mapping.values())) != EXPECTED_NUM_LABELS:
            raise ValueError(f"Label mapping at {mapping_path} contains duplicate labels.")
        return mapping

    @staticmethod
    def _load_max_length(metadata_path: Path) -> int:
        if not metadata_path.is_file():
            return DEFAULT_MAX_LENGTH

        try:
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                metadata: Any = json.load(metadata_file)
            max_length = metadata.get("max_length") if isinstance(metadata, dict) else None
            if isinstance(max_length, int) and not isinstance(max_length, bool) and max_length > 0:
                return max_length
        except (OSError, json.JSONDecodeError):
            pass
        return DEFAULT_MAX_LENGTH

    def predict(self, text: str, top_k: int = 3) -> dict[str, Any]:
        """Return the most likely intent and up to ``top_k`` scored candidates."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace only.")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if not 1 <= top_k <= EXPECTED_NUM_LABELS:
            raise ValueError(f"top_k must be between 1 and {EXPECTED_NUM_LABELS}.")

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        model_inputs = {name: tensor.to(self.device) for name, tensor in encoded.items()}

        with torch.inference_mode():
            logits = self.model(**model_inputs).logits
            if logits.ndim != 2 or logits.shape[-1] != EXPECTED_NUM_LABELS:
                raise RuntimeError(
                    "Model output shape does not match the 77 configured intent labels: "
                    f"received {tuple(logits.shape)}."
                )
            probabilities = torch.softmax(logits[0], dim=-1)
            candidate_scores, candidate_ids = torch.topk(probabilities, k=top_k)
            predicted_id = int(torch.argmax(probabilities).item())

        top_predictions = [
            {"intent": self.labels[int(class_id)], "confidence": float(score)}
            for class_id, score in zip(candidate_ids.tolist(), candidate_scores.tolist())
        ]
        return {
            "predicted_intent": self.labels[predicted_id],
            "confidence": float(probabilities[predicted_id].item()),
            "top_predictions": top_predictions,
        }


def _run_smoke_test() -> None:
    """Run the requested three local sample complaints and print predictions."""
    samples = (
        "How can I activate my card?",
        "My bank transfer has not arrived in my account.",
        "I do not recognise a transaction on my card.",
    )
    classifier = BertIntentClassifier()
    for complaint in samples:
        result = classifier.predict(complaint, top_k=3)
        print(f"Complaint: {complaint}")
        print(f"Predicted intent: {result['predicted_intent']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Top predictions:")
        for candidate in result["top_predictions"]:
            print(f"  - {candidate['intent']}: {candidate['confidence']:.4f}")
        print()


if __name__ == "__main__":
    _run_smoke_test()
