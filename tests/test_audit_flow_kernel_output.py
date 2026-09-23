import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import audit_flow_kernel_output as audit


class OutputReceiptTests(unittest.TestCase):
    def setUp(self):
        self.proof = {"selection_policy": "all_frozen_e031_gates",
                      "outside_author_selection": {"passed": True}, "flow_function_sha256": "selected-function"}
        self.receipt = {
            "external_reference_sha256": audit.reference.REFERENCE_SHA256,
            "frozen_overrides": audit.reference.FROZEN_OVERRIDES,
            "flow_reference_sha256": audit.flow.FLOW_REFERENCE_SHA256,
            "flow_function_sha256": self.proof["flow_function_sha256"],
            "flow_config": audit.flow.FLOW_CONFIG, "offline_promotion": self.proof,
            "status": "kernel_output_audited_public_score_pending", "passed": True, "flow_frames": 396,
            "effective_environment": {"BIOHUB_VALIDATOR_ENABLE": "0", **{
                "BIOHUB_" + key: str(value) for key, value in audit.reference.FROZEN_OVERRIDES.items()}},
        }

    def test_expected_receipt_passes_but_other_function_or_proof_is_rejected(self):
        audit.check_receipt(self.receipt, self.proof)
        for field, value in (("flow_function_sha256", "other"), ("offline_promotion", {}),
                             ("flow_frames", 0), ("passed", False)):
            changed = {**self.receipt, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.check_receipt(changed, self.proof)

    def test_runtime_environment_cannot_reenable_labels_or_change_overrides(self):
        for key, value in (("BIOHUB_VALIDATOR_ENABLE", "1"), ("BIOHUB_MOTION_RELINK_TIGHT_UM", "6.0")):
            changed = copy.deepcopy(self.receipt)
            changed["effective_environment"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.check_receipt(changed, self.proof)


if __name__ == "__main__":
    unittest.main()
