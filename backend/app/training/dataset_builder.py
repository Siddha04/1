"""
Builds a small instruction-tuning dataset from recently ingested chunks,
by asking the current model to draft a Q&A pair grounded in each one.
This is what "the model learns what it revised" means concretely: today's
fresh facts become tomorrow's fine-tuning examples.

Invoke from the repo root: PYTHONPATH=backend python -m app.training.dataset_builder
"""
import json
import pathlib

from app.core.llm_engine import generate
from app.core.vector_store import get_vector_store

OUT_DIR = pathlib.Path("data/training_sets")

QA_PROMPT = """Write ONE question a user might realistically ask that this
passage answers, and the correct answer using only this passage. Respond as
JSON only: {{"question": "...", "answer": "..."}}

PASSAGE:
{passage}"""


def build_dataset(sample_size: int = 200) -> pathlib.Path:
    store = get_vector_store()
    raw_docs = store.get_recent(sample_size)

    examples = []
    for text in raw_docs:
        try:
            raw_json = generate(
                query="Create a training example.",
                context=QA_PROMPT.format(passage=text),
            )
            pair = json.loads(raw_json)
            examples.append(
                {"instruction": pair["question"], "input": "", "output": pair["answer"]}
            )
        except Exception:
            continue

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "latest.jsonl"
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"[dataset_builder] wrote {len(examples)} examples to {out_path}")
    return out_path


if __name__ == "__main__":
    build_dataset()
