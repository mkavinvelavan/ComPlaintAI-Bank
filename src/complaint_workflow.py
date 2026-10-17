"""Compose local complaint classification, extraction, and internal triage.

Priority mappings are project-configured rules. The policy PDFs do not mandate
these intent-to-priority assignments for every classifier label. A predicted
intent remains a model prediction and is not a verified complaint finding.
"""

from __future__ import annotations

from typing import Any, TypedDict

from src.complaint_processor import ComplaintProcessor
from src.priority_engine import ComplaintPriorityEngine


class ComplaintWorkflowResult(TypedDict):
    """JSON-compatible internal assessment with no source complaint text."""

    classification: dict[str, Any]
    entities: list[dict[str, Any]]
    security_disclosure: dict[str, Any]
    priority_assessment: dict[str, Any]


class ComplaintWorkflow:
    """Run the existing local processor and rule-based priority engine."""

    def __init__(
        self,
        processor: ComplaintProcessor | None = None,
        priority_engine: ComplaintPriorityEngine | None = None,
    ) -> None:
        """Initialize dependencies once or reuse compatible instances."""
        self.processor = processor if processor is not None else ComplaintProcessor()
        self.priority_engine = (
            priority_engine if priority_engine is not None else ComplaintPriorityEngine()
        )

    def assess_complaint(
        self, text: str, confidence_threshold: float | None = None
    ) -> ComplaintWorkflowResult:
        """Return an internal assessment without retaining or logging ``text``.

        The optional threshold is explicit project configuration. The existing
        priority engine accepts it at construction, not in ``assess()``; for a
        per-call override this method creates a configured engine and leaves the
        retained instance untouched. ``None`` preserves its current setting.

        Raises:
            TypeError: If text is not a string.
            ValueError: If text is blank or threshold is invalid.
            RuntimeError: If the processor violates its result contract.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace only.")

        processed = self.processor.process_complaint(text)
        if not isinstance(processed, dict):
            raise RuntimeError("Complaint processor returned an invalid result; expected a dictionary.")

        engine = self.priority_engine
        if confidence_threshold is not None:
            # Delegate range/type validation to the engine's supported API.
            engine = ComplaintPriorityEngine(low_confidence_threshold=confidence_threshold)
        priority = engine.assess(processed)

        classification = {
            "predicted_intent": processed.get("predicted_intent"),
            "confidence": processed.get("confidence"),
            "top_predictions": processed.get("top_predictions", []),
            "is_model_prediction_not_verified_finding": True,
        }
        entities = processed.get("entities", [])
        disclosure = processed.get("sensitive_disclosure_detected", False)
        categories = processed.get("sensitive_categories", [])
        if not isinstance(entities, list):
            raise RuntimeError("Complaint processor returned an invalid entities list.")
        if not isinstance(disclosure, bool) or not isinstance(categories, list):
            raise RuntimeError("Complaint processor returned invalid security disclosure fields.")
        if not isinstance(priority, dict):
            raise RuntimeError("Priority engine returned an invalid assessment.")

        return {
            "classification": classification,
            "entities": entities,
            "security_disclosure": {
                "detected": disclosure,
                "categories": categories,
            },
            "priority_assessment": priority,
        }
