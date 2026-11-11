"""Relax artificial parameter locks in the reproduction COPY, preserving defaults.

Apply: python reproduction/patch_model_flexibility.py --root PROJECT --report REPORT
Audit only: add --dry-run. Refuses to change the original PiFuV2 project.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ORIGINAL = Path(r"E:\App\DaiMa\JetBrains\Project\PiFuV2")


def replace_exact(text, old, new, count=1):
    found = text.count(old)
    if new in text and (found == 0 or old in new):
        return text  # Idempotent.
    if found != count:
        raise ValueError(f"Expected {count} source fragments; found {found}: {old[:90]}")
    return text.replace(old, new)


def gate(text, name):
    constant = name.upper() + "_GATE_BALANCE_WEIGHT"
    text = replace_exact(text, "from copy import copy\n", "from copy import copy\nimport math\n")
    text = replace_exact(text, f"if self.gate_balance_weight != {constant}:",
                         "if not math.isfinite(self.gate_balance_weight) or self.gate_balance_weight < 0.0:", 2)
    text = replace_exact(text, f'f"{name} gate balance weight changed: {{self.gate_balance_weight}}"',
                         f'f"{name} gate balance weight must be finite and nonnegative: {{self.gate_balance_weight}}"')
    text = replace_exact(text, f'f"{name} gate balance weight must be {{{constant}}}"',
                         f'f"{name} gate balance weight must be finite and nonnegative: {{self.gate_balance_weight}}"')
    return text


def utility(text):
    text = replace_exact(text,
        "def run17_utility_eligible(binary_iou: float, group_macro_f1: float) -> bool:\n    return float(binary_iou) >= RUN17_MIN_BINARY_IOU and float(group_macro_f1) >= RUN17_MIN_GROUP_MACRO_F1",
        "def run17_utility_eligible(binary_iou: float, group_macro_f1: float, min_binary_iou: float = RUN17_MIN_BINARY_IOU, min_group_macro_f1: float = RUN17_MIN_GROUP_MACRO_F1) -> bool:\n    return float(binary_iou) >= min_binary_iou and float(group_macro_f1) >= min_group_macro_f1")
    text = replace_exact(text,
        'if self.min_binary_iou != RUN17_MIN_BINARY_IOU or self.min_group_macro_f1 != RUN17_MIN_GROUP_MACRO_F1:\n            raise ContractError("Run17 utility eligibility floors drifted")',
        'if not (0.0 <= self.min_binary_iou <= 1.0 and 0.0 <= self.min_group_macro_f1 <= 1.0):\n            raise ContractError("Utility eligibility floors must be finite values in [0, 1]")')
    text = replace_exact(text,
        'stats["metrics/binary_lesion_iou"], stats["metrics/group_macro_f1"]\n        )',
        'stats["metrics/binary_lesion_iou"], stats["metrics/group_macro_f1"],\n            self.min_binary_iou, self.min_group_macro_f1\n        )')
    return text


def class_weight(text):
    return replace_exact(text,
        'if self.image_cls_weight <= 0:\n            raise ContractError(f"image_cls_weight must be positive, found {self.image_cls_weight}")',
        'if not math.isfinite(self.image_cls_weight) or self.image_cls_weight < 0:\n            raise ContractError(f"image_cls_weight must be finite and nonnegative, found {self.image_cls_weight}")')


def cbam(text):
    return replace_exact(text,
        'if kernel_size not in (3, 7):\n            raise ContractError("Run33 standard CBAM spatial kernel must be 3 or 7")',
        'if kernel_size <= 0 or kernel_size % 2 == 0:\n            raise ContractError("Run33 CBAM spatial kernel must be a positive odd integer")')


def ast_evidence(before, after):
    left, right = ast.parse(before), ast.parse(after)
    # Arithmetic, forward passes, optimization, parameter initialization, and
    # checkpoint scoring methods are unchanged except utility's configurable floors.
    def methods(tree):
        return {f"{c.name}.{f.name}": ast.dump(f, include_attributes=False)
                for c in tree.body if isinstance(c, ast.ClassDef)
                for f in c.body if isinstance(f, ast.FunctionDef) and f.name != "__init__"}
    old, new = methods(left), methods(right)
    changes = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    allowed = ["UtilityEligiblePartialLabelMultiTaskValidator.get_stats"]
    if any(k not in allowed for k in changes):
        raise AssertionError(f"Unexpected method change: {changes}")
    def constants(tree):
        return {t.id: ast.dump(n.value, include_attributes=False) for n in tree.body
                if isinstance(n, ast.Assign) for t in n.targets
                if isinstance(t, ast.Name) and t.id.isupper()}
    if constants(left) != constants(right):
        raise AssertionError("Original model constants/defaults changed")
    compile(after, "patched_model.py", "exec")
    return {"original_constants_unchanged": True,
            "all_forward_and_loss_arithmetic_methods_unchanged": True,
            "changed_nonconstructor_methods": changes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    original = ORIGINAL.resolve()
    if root == original or original in root.parents or root in original.parents:
        raise ValueError("This patch utility only operates on a separate reproduction copy")
    changes = []
    tasks = []
    for run, suffix in (("lg2", "gate_balance_seed42"), ("lgx", "gate_balance_seed43"), ("lg3", "gate_balance_seed44"), ("ab1", "unguided_gate_balance_seed43")):
        tasks.append((f"tools/{run}/imaplusplus_partial_multitask_lgla_v1_{suffix}.py", lambda text, name=run: gate(text, name), "Allow any finite nonnegative gate_balance_weight; default 0.01 and formula unchanged"))
    tasks += [
        ("tools/imaplusplus_partial_multitask_utility_selection_v1.py", utility, "Use configurable eligibility floors; defaults 0.70 and 0.50 unchanged"),
        ("tools/imaplusplus_partial_multitask_v1.py", class_weight, "Allow zero image classification weight for ablations; default 0.5 unchanged"),
        ("tools/run33/imaplusplus_partial_multitask_p3p4_standard_cbam_seed43_v1.py", cbam, "Allow any positive odd spatial kernel; default 7 unchanged"),
    ]
    payloads = []
    for relative, transform, reason in tasks:
        path = root / relative
        before_bytes = path.read_bytes()
        before = before_bytes.decode("utf-8-sig").replace("\r\n", "\n")
        after = transform(before)
        evidence = ast_evidence(before, after)
        payload = after.encode("utf-8")
        changes.append({"path": relative, "reason": reason,
                        "before_sha256": hashlib.sha256(before_bytes).hexdigest(),
                        "after_sha256": hashlib.sha256(payload).hexdigest(),
                        "modified": before_bytes != payload, **evidence})
        payloads.append((path, payload))
    # Execute the actual old/new standalone functions over threshold boundaries.
    utility_path = root / "tools/imaplusplus_partial_multitask_utility_selection_v1.py"
    utility_source = utility_path.read_text(encoding="utf-8-sig")
    def get_utility(source):
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run17_utility_eligible")
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {"RUN17_MIN_BINARY_IOU": 0.70, "RUN17_MIN_GROUP_MACRO_F1": 0.50}
        exec(compile(ast.fix_missing_locations(module), "utility_probe.py", "exec"), namespace)
        return namespace["run17_utility_eligible"]
    old_utility, new_utility = get_utility(utility_source), get_utility(utility(utility_source))
    values = [-0.1, 0.0, 0.499999, 0.5, 0.699999, 0.7, 1.0]
    assert all(old_utility(x, y) == new_utility(x, y) for x in values for y in values)
    assert new_utility(0.6, 0.4, 0.6, 0.4) and not new_utility(0.6, 0.4)
    if not args.dry_run:
        for path, payload in payloads:
            path.write_bytes(payload)
    report = {"copy_root": str(root), "dry_run": args.dry_run, "modified_files": changes,
              "utility_default_boundary_cases_equal": len(values) ** 2,
              "utility_changed_threshold_case_passed": True,
              "default_behavior": "0.01 gate regularizer, 0.5 classification weight, 0.70/0.50 utility floors, CBAM kernel 7 unchanged; no tensor operation changed except utility comparisons receive the same default values",
              "preserved_structural_checks": ["AB2 has fixed fusion and no adaptive gate, so its gate balance weight must remain zero", "P3/P4 shape, class count, mask/label consistency and strict state loading", "Partial-label implementation has no semantically valid mixed-image targets, so mosaic/mixup/cutmix/copy_paste remain unsupported", "Finite/nonnegative loss weights, probability threshold ranges and positive odd convolution kernels"]}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DRY_RUN_PASS" if args.dry_run else "PATCHED", "files": len(changes), "report": str(args.report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
