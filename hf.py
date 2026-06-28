import config

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover - defensive fallback
    InferenceClient = None

MODELS = getattr(
    config,
    "HF_MODELS",
    ["meta-llama/Llama-3.1-8B-Instruct"],
)


def generate_response(prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
    key = getattr(config, "HF_API_KEY", None)
    if not key:
        return "Error: HF_API_KEY missing in config.py"

    if InferenceClient is None:
        return "Error: huggingface_hub is not installed. Install it with pip install huggingface_hub"

    last_err = None
    for model_name in MODELS:
        try:
            client = InferenceClient(model=model_name, token=key)
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_err = exc

    return (
        "Hugging Face model failed.\n"
        f"Tried models: {MODELS}\n"
        "Fix:\n"
        "1) Switch to Groq by importing groq.py in main.py OR\n"
        "2) Replace HF model in hf.py (HF_MODELS).\n"
        f"Details: {type(last_err).__name__}: {last_err}"
    )
