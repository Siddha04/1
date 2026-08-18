"""
Runs only after evaluate.py exits 0. Versions the candidate adapter and
flips the pointer llm_engine.py reads, so the next server restart is
served by the newly fine-tuned model — trivially revertible, since this
script keeps the last 3 versions around.

Invoke from the repo root: PYTHONPATH=backend python -m app.training.promote_adapter
"""
import datetime as dt
import pathlib
import shutil

from app.config import get_settings

settings = get_settings()


def main():
    candidate = pathlib.Path("models/adapters/candidate")
    if not candidate.exists():
        raise SystemExit("No candidate adapter found — run train_lora.py first.")

    version = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    versioned_dir = pathlib.Path(settings.ADAPTER_DIR) / version
    shutil.copytree(candidate, versioned_dir)

    pathlib.Path(settings.ACTIVE_ADAPTER_FILE).write_text(str(versioned_dir))
    print(f"[promote_adapter] promoted {versioned_dir} to active")

    # keep only the 3 most recent versions
    versions = sorted(p for p in pathlib.Path(settings.ADAPTER_DIR).iterdir() if p.is_dir())
    for old in versions[:-3]:
        shutil.rmtree(old, ignore_errors=True)


if __name__ == "__main__":
    main()
