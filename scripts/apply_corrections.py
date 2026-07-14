"""Apply a correction file to a pipeline-output event log (FR-019).

Usage:
    python scripts/apply_corrections.py \
        --pipeline-output outputs/verification/clip_0007_events.json \
        --corrections outputs/verification/clip_0007_corrections.json \
        --out outputs/verification/clip_0007_corrected.json

If --out is omitted, defaults to "<pipeline-output-stem>_corrected.json" next
to the pipeline-output file. Refuses to run if the resolved output path would
overwrite either input file: corrections are always layered as a new,
separate file, never applied in place (design 5.14) — this is enforced here,
not just left to convention.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.verification.correction import apply_corrections, write_corrected_output  # noqa: E402
from src.verification.correction_schema import CorrectionFile  # noqa: E402
from src.verification.events import PipelineOutputEvents  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply a correction file to a pipeline output (FR-019)."
    )
    ap.add_argument("--pipeline-output", required=True,
                     help="path to the PipelineOutputEvents JSON")
    ap.add_argument("--corrections", required=True, help="path to the CorrectionFile JSON")
    ap.add_argument("--out", default=None,
                     help="output path (default: <pipeline-output>_corrected.json)")
    args = ap.parse_args()

    pipeline_path = Path(args.pipeline_output)
    corrections_path = Path(args.corrections)
    if not pipeline_path.exists():
        raise SystemExit(f"pipeline output not found: {pipeline_path}")
    if not corrections_path.exists():
        raise SystemExit(f"correction file not found: {corrections_path}")

    out_path = (
        Path(args.out) if args.out
        else pipeline_path.with_name(f"{pipeline_path.stem}_corrected.json")
    )
    if out_path.resolve() in {pipeline_path.resolve(), corrections_path.resolve()}:
        raise SystemExit(
            f"refusing to write corrected output over an input file: {out_path}"
        )

    pipeline_output = PipelineOutputEvents.from_json(pipeline_path)
    correction_file = CorrectionFile.from_json(corrections_path)

    corrected = apply_corrections(pipeline_output, correction_file)
    write_corrected_output(corrected, out_path)

    print(f"applied {len(correction_file.corrections)} correction(s) from {corrections_path}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
