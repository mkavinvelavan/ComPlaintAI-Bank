"""Coordinate local complaint intent classification and entity extraction."""

from __future__ import annotations

from typing import Any, TypedDict

from src.bert_intent_classifier import BertIntentClassifier
from src.entity_extractor import SpacyEntityExtractor
from src.intent_corrections import correct_card_delivery_intent


class ComplaintProcessingResult(TypedDict):
    """Combined prediction and extraction result; does not include complaint text."""

    predicted_intent: str
    confidence: float | None
    top_predictions: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    sensitive_disclosure_detected: bool
    sensitive_categories: list[str]
    model_prediction: dict[str, Any]
    intent_correction: dict[str, Any]


class ComplaintProcessor:
    """Reuse the local BERT classifier and spaCy extractor for each complaint.

    The processor is stateless between calls: it retains the component instances
    but does not store complaint text or results. It performs no persistence,
    prioritization, routing, logging, or external service calls.
    """

    def __init__(
        self,
        classifier: BertIntentClassifier | None = None,
        entity_extractor: SpacyEntityExtractor | None = None,
    ) -> None:
        """Initialize both components once, or accept compatible instances.

        Optional component arguments make the boundary easy to isolate in tests
        and allow callers to reuse already initialized component instances.
        """
        self.classifier = classifier if classifier is not None else BertIntentClassifier()
        self.entity_extractor = (
            entity_extractor if entity_extractor is not None else SpacyEntityExtractor()
        )

    def process_complaint(self, text: str) -> ComplaintProcessingResult:
        """Classify and extract supported entities from complaint text.

        Args:
            text: A non-empty complaint string. It is passed directly to the
                two local components and is not logged or included in the result.

        Raises:
            TypeError: If ``text`` is not a string.
            ValueError: If ``text`` is empty or whitespace only.
            RuntimeError: If a component returns a result outside its contract.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace only.")

        classification = self.classifier.predict(text, top_k=3)
        extraction = self.entity_extractor.extract(text)

        required_classification_fields = {
            "predicted_intent",
            "confidence",
            "top_predictions",
        }
        if not isinstance(classification, dict) or not required_classification_fields.issubset(
            classification
        ):
            raise RuntimeError(
                "BERT classifier returned an invalid result; expected predicted_intent, "
                "confidence, and top_predictions."
            )

        # Keep the complete untouched model output for diagnosis. A corrected
        # intent has no calibrated BERT score, so confidence is omitted for the
        # corrected label and the raw model score remains in model_prediction.
        model_prediction = {
            "predicted_intent": classification["predicted_intent"],
            "confidence": classification["confidence"],
            "top_predictions": classification["top_predictions"],
        }
        corrected_intent = correct_card_delivery_intent(text)
        correction_applied = (
            corrected_intent is not None
            and corrected_intent != classification["predicted_intent"]
        )
        final_intent = corrected_intent or classification["predicted_intent"]
        if not isinstance(extraction, dict) or not isinstance(extraction.get("entities"), list):
            raise RuntimeError(
                "Entity extractor returned an invalid result; expected an entities list."
            )

        # The existing extractor provides these fields. Defaults preserve a
        # stable combined schema if a compatible extractor omits optional flags.
        security_flags = extraction.get("security_flags", {})
        if not isinstance(security_flags, dict):
            raise RuntimeError("Entity extractor returned invalid security_flags.")
        disclosure_detected = security_flags.get("credential_disclosed", False)
        sensitive_categories = security_flags.get("credential_categories", [])
        if not isinstance(disclosure_detected, bool) or not isinstance(
            sensitive_categories, list
        ):
            raise RuntimeError("Entity extractor returned invalid credential disclosure fields.")

        return {
            "predicted_intent": final_intent,
            "confidence": None if correction_applied else classification["confidence"],
            "top_predictions": classification["top_predictions"],
            "entities": extraction["entities"],
            "sensitive_disclosure_detected": disclosure_detected,
            "sensitive_categories": sensitive_categories,
            "model_prediction": model_prediction,
            "intent_correction": {
                "applied": correction_applied,
                "rule": corrected_intent if correction_applied else None,
            },
        }
