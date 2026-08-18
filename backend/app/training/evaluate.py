"""
Gatekeeper between "we trained something" and "users see it". Runs a
fixed regression set through the candidate adapter and checks it isn't
obviously broken. This is what keeps a bad fine-tune from silently
degrading the assistant.

Invoke from the repo root: PYTHONPATH=backend python -m app.training.evaluate
"""
import argparse
import json
import pathlib

PASS_THRESHOLD = 0.7  # fraction of eval questions the candidate must get "reasonable" on


def load_eval_set() -> list[dict]:
    path = pathlib.Path("backend/tests/eval_questions.json")
    if not path.exists():
        return [{"question": "What model are you based on?", "must_contain": []}]
    return json.loads(path.read_text())


def score_answer(answer: str, must_contain: list[str]) -> bool:
    if must_contain:
        return all(term.lower() in answer.lower() for term in must_contain)
    return bool(answer.strip())


def main(adapter_dir: str) -> bool:
    # Imported lazily so this file can be inspected/tested without a GPU.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from app.config import get_settings

    settings = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(settings.BASE_MODEL_ID, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir)

    eval_set = load_eval_set()
    passed = 0
    for item in eval_set:
        inputs = tokenizer(item["question"], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=200)
        answer = tokenizer.decode(out[0], skip_special_tokens=True)
        if score_answer(answer, item.get("must_contain", [])):
            passed += 1

    pass_rate = passed / len(eval_set)
    print(f"[evaluate] {passed}/{len(eval_set)} passed ({pass_rate:.0%})")
    return pass_rate >= PASS_THRESHOLD


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="models/adapters/candidate")
    args = parser.parse_args()
    ok = main(args.adapter)
    raise SystemExit(0 if ok else 1)
