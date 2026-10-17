"""Unit tests for deterministic complaint entity extraction."""

import unittest

from src.entity_extractor import SpacyEntityExtractor


class SpacyEntityExtractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = SpacyEntityExtractor()

    def test_extracts_reference_amount_date_and_card_suffix(self) -> None:
        text = (
            "I was charged AED 125.50 on 17/10/2026 for transaction reference "
            "TXN-AB1234. My card ending in 4821 was used."
        )
        result = self.extractor.extract(text)
        found = {entity["entity_type"]: entity["text"] for entity in result["entities"]}

        self.assertEqual(found["TRANSACTION_REFERENCE"], "TXN-AB1234")
        self.assertEqual(found["MONEY_AMOUNT"], "AED 125.50")
        self.assertEqual(found["DATE"], "17/10/2026")
        self.assertEqual(found["CARD_LAST4"], "4821")
        self.assertFalse(result["security_flags"]["credential_disclosed"])

    def test_extracts_account_suffix(self) -> None:
        result = self.extractor.extract("The account ending in 1234 is not mine.")
        self.assertEqual(
            result["entities"],
            [
                {
                    "text": "1234",
                    "entity_type": "ACCOUNT_LAST4",
                    "start": 22,
                    "end": 26,
                }
            ],
        )

    def test_recognizes_written_dates_and_currency_symbols(self) -> None:
        text = "On October 17, 2026 I paid USD 40.00 and ₹500."
        result = self.extractor.extract(text)
        found = [(entity["entity_type"], entity["text"]) for entity in result["entities"]]
        self.assertIn(("DATE", "October 17, 2026"), found)
        self.assertIn(("MONEY_AMOUNT", "USD 40.00"), found)
        self.assertIn(("MONEY_AMOUNT", "₹500"), found)

    def test_returns_no_entities_for_unmatched_text(self) -> None:
        result = self.extractor.extract("Please explain how the service works.")
        self.assertEqual(result["entities"], [])
        self.assertEqual(
            result["security_flags"],
            {"credential_disclosed": False, "credential_categories": []},
        )

    def test_rejects_empty_and_non_string_input(self) -> None:
        for text in ("", " \t\n"):
            with self.subTest(text=repr(text)):
                with self.assertRaises(ValueError):
                    self.extractor.extract(text)
        with self.assertRaises(TypeError):
            self.extractor.extract(None)  # type: ignore[arg-type]

    def test_credential_values_are_never_returned(self) -> None:
        text = (
            "My OTP is 739184, my PIN: 4821, my CVV is 123, "
            "and my password is hunter2."
        )
        result = self.extractor.extract(text)

        returned_text = " ".join(entity["text"] for entity in result["entities"])
        self.assertTrue(
            all(secret not in returned_text for secret in ("739184", "4821", "123", "hunter2")),
            "credential values must not appear in extracted entities",
        )
        self.assertTrue(result["security_flags"]["credential_disclosed"])
        self.assertEqual(
            result["security_flags"]["credential_categories"],
            ["CVV", "OTP", "PASSWORD", "PIN"],
        )

    def test_detects_labeled_passcodes_and_verification_codes(self) -> None:
        text = (
            "Passcode: SyntheticPass44! and I cannot sign in. "
            "Verification code is 882211, then access failed."
        )
        result = self.extractor.extract(text)
        returned_text = " ".join(entity["text"] for entity in result["entities"])

        self.assertTrue(result["security_flags"]["credential_disclosed"])
        self.assertEqual(
            result["security_flags"]["credential_categories"],
            ["OTP", "PASSCODE"],
        )
        self.assertTrue(
            all(value not in returned_text for value in ("SyntheticPass44", "882211")),
            "credential values must not appear in extracted entities",
        )

    def test_entity_offsets_reference_original_text(self) -> None:
        text = "Transaction ref TXN-77; card ending in 4821."
        result = self.extractor.extract(text)
        for entity in result["entities"]:
            self.assertEqual(
                text[entity["start"] : entity["end"]], entity["text"]
            )
        reference = next(
            entity
            for entity in result["entities"]
            if entity["entity_type"] == "TRANSACTION_REFERENCE"
        )
        self.assertEqual(reference["start"], text.index("TXN-77"))
        self.assertEqual(reference["end"], text.index("TXN-77") + len("TXN-77"))

    def test_overlapping_and_duplicate_candidates_are_resolved(self) -> None:
        duplicate = {
            "text": "4821",
            "entity_type": "CARD_LAST4",
            "start": 10,
            "end": 14,
        }
        overlapping = {
            "text": "10 4821",
            "entity_type": "DATE",
            "start": 7,
            "end": 14,
        }
        resolved = self.extractor._resolve_overlaps(
            [duplicate, duplicate.copy(), overlapping]
        )
        self.assertEqual(resolved, [duplicate])


if __name__ == "__main__":
    unittest.main()
