"""Append read-only-by-default GRU/BiLSTM section to canonical Notebook 11.

Run from project root: python tools/install_recurrent_extension.py --apply
Default (without --apply) is only a preflight; no notebook changes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

MARKER = "## 11. GRU and BiLSTM extension"


def md(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source: str, cell_id: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


CELLS = [
    md("""## 11. GRU and BiLSTM extension

The earlier sections complete the 360-fold LSTM and TCN study. This extension adds GRU and bidirectional LSTM (BiLSTM) without changing those archived results. Both recurrent models use the same four layer configurations as LSTM. Each is evaluated across three forecast horizons, five feature families and three chronological folds: another 360 evaluations in total.

The 13-step sequence contains measurements from the hour preceding the forecast issue time. BiLSTM reads this **historical** sequence in both directions; it does not access observations after the issue time. The outer validation periods, inner early-stopping split, 60-minute label embargo, seed, optimizer and evaluation metric remain the same as the earlier neural campaign.

For consistency, F0 uses the original V1 train-only scaling code. F1–F4 use the V2 scaling code, including the year-safe treatment for a constant year in the inner training fold. The new results have their own checkpoint directories. The original LSTM and TCN files are not updated.

**Training remains off during Run All.** To begin a small batch later, change the explicit extension switch in the following cell and run that cell only. The June–August 2019 terminal period is still excluded.
""", "ext-a1c2d3e4"),
    code("""# Read the GRU/BiLSTM archive without starting any training by default.
EXT_RUN_BATCH = False
EXT_MAX_NEW_FITS = 1
EXT_MODELS = ("GRU", "BiLSTM")

from src.revised_recurrent_extension import (
    EXPECTED_FITS as EXT_EXPECTED_FITS,
    F0_ARCHIVE as EXT_F0_ARCHIVE,
    V2_ARCHIVE as EXT_V2_ARCHIVE,
    read_progress as read_recurrent_extension,
    complete_rankings as rank_recurrent_extension,
    run_batch as run_recurrent_extension,
)

assert M2_RUN_PILOTS is False
assert M3_RUN_LSTM is False and M3_RUN_TCN is False
assert M4_RUN_ONE is False and M5_RUN_BATCH is False and M6_RUN_BATCH is False
assert len(m6_final_progress) == 360, "Check the completed original CV archive first"

ext_report = run_recurrent_extension(
    root=PROJECT_ROOT,
    payload_factory=m5_payload_for,
    models=EXT_MODELS,
    max_new_fits=EXT_MAX_NEW_FITS,
    enabled=EXT_RUN_BATCH,
)
print("GRU/BiLSTM DEVELOPMENT CV:", ext_report["after"], "/", EXT_EXPECTED_FITS)
if not EXT_RUN_BATCH:
    assert ext_report["new_fits"] == 0
    print("GRU/BiLSTM BATCH: DISABLED — no training")
""", "ext-b5f6a7b8"),
    md("""### Running the additional models

Start with one GRU fit and one BiLSTM fit, in separate batches. After checking that both finish and their checkpoints can be read, continue in batches of about 10–15 fits. The batch runner automatically skips work already archived.

To run the first GRU fit, change `EXT_RUN_BATCH` to `True`, set `EXT_MODELS = ("GRU",)` and `EXT_MAX_NEW_FITS = 1` in the previous cell, then execute **only that cell**. Repeat with `EXT_MODELS = ("BiLSTM",)` for the first BiLSTM fit. Restore `EXT_RUN_BATCH = False` immediately after each batch and save the notebook. Never select Run All while a training switch is on.

Both new archive directories are under `results/development_only/`. A completed fold is recorded as JSON only after the selected number of epochs has been refitted and its outer-fold score has been calculated. If a run is interrupted, its already-completed files can be reused. The new archive has 360 intended records; combining it with the original 360 gives 720 evaluations across all four architectures.
""", "ext-c9d0e1f2"),
    code("""# Read-only comparison; report results only when enough new folds are archived.
from IPython.display import display

ext_scores = read_recurrent_extension(PROJECT_ROOT)
ext_rankings = rank_recurrent_extension(PROJECT_ROOT)
print(f"GRU/BiLSTM completed: {len(ext_scores)}/360 checkpoint records")

if len(ext_scores):
    progress = (ext_scores.groupby(["model", "horizon_minutes"])
                .size().rename("completed_folds").reset_index())
    display(progress)

if not ext_rankings.empty:
    ext_winners = ext_rankings.loc[ext_rankings["rank"] == 1]
    print("Completed GRU/BiLSTM feature-family comparisons (development only)")
    display(ext_winners[["model", "horizon_minutes", "feature_family",
                         "candidate_id", "cv_mae_kw", "cv_rmse_kw"]].round(4))
else:
    print("Full three-fold candidate groups are not available yet.")

if len(ext_scores) == 360:
    assert len(ext_rankings) == 120, "Expected 120 GRU/BiLSTM candidate rankings"
    all_rankings = pd.concat([cv_rankings, ext_rankings], ignore_index=True)
    assert len(all_rankings) == 240
    winners = (all_rankings.sort_values(
        ["model", "horizon_minutes", "cv_mae_kw", "feature_family", "config_index"],
        kind="stable")
        .drop_duplicates(["model", "horizon_minutes"])
        .sort_values(["horizon_minutes", "model"]))
    print("Four-architecture development-CV comparison")
    display(winners[["model", "horizon_minutes", "feature_family",
                     "candidate_id", "cv_mae_kw", "cv_rmse_kw"]].round(4))
else:
    print("Four-architecture comparison will be produced after all 360 additional folds finish.")
""", "ext-a3b4c5d6"),
    md("""### Status of this extension

The original 360 LSTM/TCN folds are complete and remain the reference comparison. The additional GRU/BiLSTM campaign is treated as a separate development-only experiment until its full set of records is available. It is too early to say whether either new model improves the results. Final architecture selection must use the completed three-fold development scores, not the untouched terminal holdout.
""", "ext-e7f8a9b0"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Make a dated backup and append section")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    target = root / "notebooks/11_revised_neural_models.ipynb"
    extension = root / "src/revised_recurrent_extension.py"
    if not target.is_file() or not extension.is_file():
        raise SystemExit("Run from project root after extracting src/ and tools/ from the extension ZIP")
    book = json.loads(target.read_text(encoding="utf-8"))
    if book.get("nbformat") != 4 or not isinstance(book.get("cells"), list):
        raise SystemExit("Unexpected notebook format; no changes made")
    if any(MARKER in "".join(cell.get("source", [])) for cell in book["cells"]):
        print("GRU/BiLSTM extension already installed; no changes made")
        return
    if not any("NEURAL DEVELOPMENT CV" in "".join(cell.get("source", []))
               for cell in book["cells"]):
        raise SystemExit("Notebook 11 final development-CV audit not found; no changes made")
    print(f"Notebook: {target}")
    print(f"Existing cells: {len(book['cells'])}; new cells: {len(CELLS)}")
    if not args.apply:
        print("DRY RUN — notebook not modified. Re-run with --apply to install.")
        return
    backup = target.with_name(target.stem + "_before_recurrent_extension_backup.ipynb")
    if backup.exists():
        raise SystemExit(f"Backup already exists: {backup}; review before retrying")
    shutil.copy2(target, backup)
    book["cells"].extend(CELLS)
    target.write_text(json.dumps(book, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Installed: {target}")
    print(f"Backup: {backup} (do not stage the backup unless intended)")
    print("Training remains disabled; run the notebook to verify read-only behaviour.")


if __name__ == "__main__":
    main()
