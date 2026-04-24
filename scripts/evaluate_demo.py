import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.prediction_service import predict_mri


DATASET_TEST_DIR = ROOT / "dataset" / "test"
WEB_SAMPLES_DIR = ROOT / "web_samples"
REPORTS_DIR = ROOT / "reports"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DEFAULT_REPORT_PATH = REPORTS_DIR / "demo_evaluation_report.md"


class LocalFileStorage:
    def __init__(self, path):
        self.path = Path(path)
        self.filename = self.path.name

    def read(self):
        return self.path.read_bytes()


def evaluate_dataset(limit_per_class):
    rows = []
    for expected_label in CLASS_NAMES:
        class_dir = DATASET_TEST_DIR / expected_label
        paths = sorted(class_dir.glob("*.jpg"))[:limit_per_class]
        for path in paths:
            result = predict_mri(LocalFileStorage(path))
            rows.append(build_row("dataset", path, expected_label, result))
    return rows


def evaluate_web_samples():
    if not WEB_SAMPLES_DIR.exists():
        return []

    web_expectations = {
        "meningioma_web.png": "meningioma",
        "pituitary_web.jpg": "pituitary",
        "normal_web.jpg": "notumor",
        "glioma_web.jpg": "glioma",
    }

    rows = []
    for filename, expected_label in web_expectations.items():
        path = WEB_SAMPLES_DIR / filename
        if not path.exists():
            continue

        try:
            Image.open(path).verify()
        except Exception:
            continue

        result = predict_mri(LocalFileStorage(path))
        rows.append(build_row("web", path, expected_label, result))
    return rows


def build_row(source, path, expected_label, result):
    assessment = result.get("assessment", {})
    return {
        "source": source,
        "file": str(path.relative_to(ROOT)),
        "expected": expected_label,
        "predicted": result["prediction"],
        "confidence": float(result["confidence"]),
        "confidence_band": assessment.get("confidence_band", "unknown"),
        "runner_up": assessment.get("runner_up"),
        "margin": assessment.get("margin"),
    }


def summarize(rows):
    correct = sum(1 for row in rows if row["expected"] == row["predicted"])
    accuracy = (correct / len(rows) * 100) if rows else 0.0
    by_class = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = Counter()

    for row in rows:
        by_class[row["expected"]]["total"] += 1
        if row["expected"] == row["predicted"]:
            by_class[row["expected"]]["correct"] += 1
        confusion[(row["expected"], row["predicted"])] += 1

    return {
        "samples": len(rows),
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "by_class": dict(by_class),
        "confusion": {
            f"{expected}->{predicted}": count
            for (expected, predicted), count in sorted(confusion.items())
        },
        "mismatches": [row for row in rows if row["expected"] != row["predicted"]],
    }


def format_summary_lines(title, summary):
    lines = [f"\n{title}", "-" * len(title)]
    if not summary["samples"]:
        lines.append("No samples available.")
        return lines

    lines.append(f"Samples: {summary['samples']}")
    lines.append(f"Accuracy: {summary['accuracy']:.2f}%")
    lines.append("")
    lines.append("Per-class:")
    for class_name in CLASS_NAMES:
        item = summary["by_class"].get(class_name)
        if item and item["total"]:
            lines.append(f"  {class_name}: {item['correct']}/{item['total']}")

    lines.append("")
    lines.append("Mismatches:")
    if not summary["mismatches"]:
        lines.append("  none")
    else:
        for row in summary["mismatches"]:
            lines.append(
                f"  {row['file']}: expected {row['expected']}, predicted "
                f"{row['predicted']} ({row['confidence'] * 100:.2f}%, "
                f"{row['confidence_band']}, runner-up {row['runner_up']}, "
                f"margin {row['margin']}%)"
            )

    lines.append("")
    lines.append("Confusion pairs:")
    for pair, count in summary["confusion"].items():
        lines.append(f"  {pair}: {count}")
    return lines


def build_markdown_report(dataset_summary, web_summary, dataset_limit):
    lines = [
        "# Brain Tumor Classification Demo Evaluation",
        "",
        "Educational use only. This report is not a clinical validation.",
        "",
        "## Run Settings",
        "",
        f"- Dataset samples per class: {dataset_limit}",
        f"- Public web samples used: {web_summary['samples']}",
        "",
        "## Dataset Spot-check",
        "",
        f"- Samples: {dataset_summary['samples']}",
        f"- Accuracy: {dataset_summary['accuracy']:.2f}%",
        "",
        "Per-class results:",
    ]

    for class_name in CLASS_NAMES:
        item = dataset_summary["by_class"].get(class_name)
        if item and item["total"]:
            lines.append(f"- {class_name}: {item['correct']}/{item['total']}")

    lines.extend(["", "Dataset mismatches:"])
    if not dataset_summary["mismatches"]:
        lines.append("- none")
    else:
        for row in dataset_summary["mismatches"]:
            lines.append(
                f"- {row['file']}: expected `{row['expected']}`, predicted "
                f"`{row['predicted']}` at {row['confidence'] * 100:.2f}%"
            )

    lines.extend(
        [
            "",
            "## Public Web Samples",
            "",
            f"- Samples: {web_summary['samples']}",
            f"- Accuracy: {web_summary['accuracy']:.2f}%",
            "",
            "Web mismatches:",
        ]
    )

    if not web_summary["mismatches"]:
        lines.append("- none")
    else:
        for row in web_summary["mismatches"]:
            lines.append(
                f"- {row['file']}: expected `{row['expected']}`, predicted "
                f"`{row['predicted']}` at {row['confidence'] * 100:.2f}% "
                f"({row['confidence_band']})"
            )

    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            "- Strong on the sampled in-project test images.",
            "- Still unreliable on unseen public MRI samples.",
            "- Safe to present only as an educational demo, not a clinical tool.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report_path, dataset_summary, web_summary, dataset_limit):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_markdown_report(dataset_summary, web_summary, dataset_limit),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a lightweight demo-readiness evaluation."
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=10,
        help="Number of dataset test images to sample per class.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown file path for the generated evaluation report.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary in addition to the text report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_rows = evaluate_dataset(args.limit_per_class)
    web_rows = evaluate_web_samples()
    dataset_summary = summarize(dataset_rows)
    web_summary = summarize(web_rows)

    print("Brain Tumor Classification Demo Evaluation")
    print("Educational use only. This report is not a clinical validation.")
    for line in format_summary_lines("Dataset spot-check", dataset_summary):
        print(line)
    for line in format_summary_lines("Public web samples", web_summary):
        print(line)

    write_report(args.report_path, dataset_summary, web_summary, args.limit_per_class)
    print(f"\nSaved report: {args.report_path.relative_to(ROOT)}")

    if args.json:
        print(
            json.dumps(
                {
                    "dataset": dataset_summary,
                    "web": web_summary,
                    "report_path": str(args.report_path.relative_to(ROOT)),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
