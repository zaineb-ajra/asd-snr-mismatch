"""Tables of the internal controls of a model (not part of the study).

Usage::

    python scripts/controls_report.py --model osscl > results/verification/osscl_tables.md

Reads ``<output_root>/verification/<model>/internal/<condition>/seed<k>/`` written
by ``python -m noisemismatch.controls``.
"""

from __future__ import annotations

import argparse

from mobilenetv2_report import controls

from noisemismatch.config import load_paths


def main(argv: list[str] | None = None) -> int:
    """Prints the Markdown tables."""
    parser = argparse.ArgumentParser(description="Internal-control tables.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition", default="0dB")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    root = load_paths().output_root / "verification" / args.model / "internal"
    lines = [
        "**Code validation only. The AUC values below (training at 0 dB, one seed) "
        "are implementation checks; they are not results of the study, not even "
        "partial ones.**",
        "",
        *controls(root / args.condition / f"seed{args.seed}"),
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
