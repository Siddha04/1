"""
Generation layer. Loads the base model once (4-bit quantized on GPU, plain
fp32 on CPU) and applies whichever LoRA adapter is currently marked
"active" by the training pipeline's promote_adapter.py. This is the
literal "model that updates itself": each CI/CD training run that passes
evaluation flips which adapter gets loaded on the next restart.
"""
import pathlib
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app.config import get_settings

settings = get_settings()

SYSTEM_PROMPT = """You are a precise personal research assistant.
Answer using ONLY the CONTEXT provided below. If the context does not
contain the answer, say so plainly instead of guessing. Always mention
how recent the information is when it's time-sensitive (prices, scores,
news)."""

_lock = threading.Lock()
_tokenizer = None
_model = None


def _active_adapter_path() -> str | None:
    p = pathlib.Path(settings.ACTIVE_ADAPTER_FILE)
    if p.exists():
        path = p.read_text().strip()
        return path or None
    return None


def _load_model():
    global _tokenizer, _model
    with _lock:
        if _model is not None:
            return

        _tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_ID)
        _tokenizer.pad_token = _tokenizer.pad_token or _tokenizer.eos_token

        if torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base = AutoModelForCausalLM.from_pretrained(
                settings.BASE_MODEL_ID, quantization_config=quant_config, device_map="auto"
            )
        else:
            # CPU fallback — works for development, but a 7B model will be
            # slow (seconds per token). Use a GPU for real usage.
            base = AutoModelForCausalLM.from_pretrained(
                settings.BASE_MODEL_ID, torch_dtype=torch.float32
            )

        adapter_path = _active_adapter_path()
        if adapter_path:
            from peft import PeftModel

            _model = PeftModel.from_pretrained(base, adapter_path)
            print(f"[llm_engine] loaded fine-tuned adapter: {adapter_path}")
        else:
            _model = base
            print("[llm_engine] no promoted adapter yet — serving the base model")


def generate(query: str, context: str) -> str:
    if _model is None:
        _load_model()

    user_content = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}"
    messages = [{"role": "user", "content": user_content}]

    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=3072
    ).to(_model.device)

    output = _model.generate(
        **inputs,
        max_new_tokens=settings.LLM_MAX_TOKENS,
        temperature=settings.LLM_TEMPERATURE,
        do_sample=settings.LLM_TEMPERATURE > 0,
        pad_token_id=_tokenizer.eos_token_id,
    )
    generated = output[0][inputs["input_ids"].shape[1] :]
    return _tokenizer.decode(generated, skip_special_tokens=True).strip()
