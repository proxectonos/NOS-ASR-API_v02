import json
import os
from typing import Any

import torch
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
    pipeline,
)


def read_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_cache_dir(models_root: str, model_id: str) -> str:
    target = os.path.join(models_root, model_id)
    os.makedirs(target, exist_ok=True)
    return target


def _get_device(use_cuda: bool) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _language_name(config_data: dict, lang: str) -> str:
    return config_data.get("languages", {}).get(lang, lang)


def _base_model_record(
    *,
    backend: str,
    model_id: str,
    model_type: str,
    lang: str,
    config_data: dict,
    sampling_rate: int,
    hf_repo: str,
    device: torch.device,
    has_lm: bool = False,
    options: dict[str, Any] | None = None,
) -> dict:
    return {
        "backend": backend,
        "model_id": model_id,
        "model_type": model_type,
        "device": device,
        "lang": lang,
        "language": _language_name(config_data, lang),
        "sampling_rate": sampling_rate,
        "has_lm": has_lm,
        "hf_repo": hf_repo,
        "options": options or {},
    }


def load_models(config_data: dict, models_root: str, use_cuda: bool):
    device = _get_device(use_cuda)
    loaded = {}
    defaults_by_lang = {}

    for entry in config_data.get("models", []):
        if not entry.get("load", False):
            continue

        model_id = entry["model_id"]
        lang = entry["lang"]
        hf_repo = entry["hf_repo"]
        sampling_rate = entry.get("sampling_rate", 16000)
        model_type = entry.get("model_type", "wav2vec2_lm")
        cache_dir = _resolve_cache_dir(models_root, model_id)

        print(
            f"[loader] loading {model_id} "
            f"({model_type}) from {hf_repo} -> {cache_dir}"
        )

        if model_type == "wav2vec2_lm":
            try:
                processor = Wav2Vec2ProcessorWithLM.from_pretrained(
                    hf_repo,
                    cache_dir=cache_dir,
                )
                has_lm = True
            except Exception as e:
                print(
                    f"[loader] LM processor failed ({e}); "
                    "falling back to plain processor"
                )
                processor = Wav2Vec2Processor.from_pretrained(
                    hf_repo,
                    cache_dir=cache_dir,
                )
                has_lm = False

            model = Wav2Vec2ForCTC.from_pretrained(
                hf_repo,
                cache_dir=cache_dir,
            ).to(device).eval()

            record = _base_model_record(
                backend="ctc",
                model_id=model_id,
                model_type=model_type,
                lang=lang,
                config_data=config_data,
                sampling_rate=sampling_rate,
                hf_repo=hf_repo,
                device=device,
                has_lm=has_lm,
            )
            record["processor"] = processor
            record["model"] = model
            loaded[model_id] = record

        elif model_type == "wav2vec2":
            processor = Wav2Vec2Processor.from_pretrained(
                hf_repo,
                cache_dir=cache_dir,
            )

            model = Wav2Vec2ForCTC.from_pretrained(
                hf_repo,
                cache_dir=cache_dir,
            ).to(device).eval()

            record = _base_model_record(
                backend="ctc",
                model_id=model_id,
                model_type=model_type,
                lang=lang,
                config_data=config_data,
                sampling_rate=sampling_rate,
                hf_repo=hf_repo,
                device=device,
                has_lm=False,
            )
            record["processor"] = processor
            record["model"] = model
            loaded[model_id] = record

        elif model_type == "wav2vec2_bert":
            processor = AutoProcessor.from_pretrained(
                hf_repo,
                cache_dir=cache_dir,
            )

            model = AutoModelForCTC.from_pretrained(
                hf_repo,
                cache_dir=cache_dir,
            ).to(device).eval()

            record = _base_model_record(
                backend="ctc",
                model_id=model_id,
                model_type=model_type,
                lang=lang,
                config_data=config_data,
                sampling_rate=sampling_rate,
                hf_repo=hf_repo,
                device=device,
                has_lm=False,
            )
            record["processor"] = processor
            record["model"] = model
            loaded[model_id] = record

        elif model_type == "whisper":
            options = entry.get("options", {})
            pipe_device = 0 if device.type == "cuda" else -1
            torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

            asr_pipeline = pipeline(
                task="automatic-speech-recognition",
                model=hf_repo,
                cache_dir=cache_dir,
                device=pipe_device,
                torch_dtype=torch_dtype,
            )

            record = _base_model_record(
                backend="whisper",
                model_id=model_id,
                model_type=model_type,
                lang=lang,
                config_data=config_data,
                sampling_rate=sampling_rate,
                hf_repo=hf_repo,
                device=device,
                has_lm=False,
                options=options,
            )
            record["pipeline"] = asr_pipeline
            loaded[model_id] = record

        else:
            raise ValueError(
                f"Unsupported model_type '{model_type}' for model '{model_id}'"
            )

        if entry.get("default") or lang not in defaults_by_lang:
            defaults_by_lang[lang] = model_id

    return loaded, defaults_by_lang


def _move_inputs_to_device(inputs, device: torch.device) -> dict:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _transcribe_ctc(loaded_model: dict, audio) -> str:
    processor = loaded_model["processor"]
    model = loaded_model["model"]
    device = loaded_model["device"]
    sampling_rate = loaded_model["sampling_rate"]

    inputs = processor(
        audio,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )

    inputs = _move_inputs_to_device(inputs, device)

    with torch.no_grad():
        logits = model(**inputs).logits

    if loaded_model["has_lm"]:
        decoded = processor.batch_decode(logits.cpu().numpy())
        text = decoded.text[0] if hasattr(decoded, "text") else decoded["text"][0]
    else:
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]

    return text.strip()


def _transcribe_whisper(loaded_model: dict, audio) -> str:
    asr_pipeline = loaded_model["pipeline"]
    sampling_rate = loaded_model["sampling_rate"]
    options = loaded_model.get("options", {})

    generate_kwargs = {
        "language": options.get("language", loaded_model["lang"]),
        "task": options.get("task", "transcribe"),
    }

    if "max_new_tokens" in options:
        generate_kwargs["max_new_tokens"] = options["max_new_tokens"]

    call_kwargs = {
        "generate_kwargs": generate_kwargs,
        "return_timestamps": options.get("return_timestamps", False),
    }

    if "chunk_length_s" in options:
        call_kwargs["chunk_length_s"] = options["chunk_length_s"]

    if "stride_length_s" in options:
        call_kwargs["stride_length_s"] = options["stride_length_s"]

    if "batch_size" in options:
        call_kwargs["batch_size"] = options["batch_size"]

    with torch.no_grad():
        result = asr_pipeline(
            {
                "raw": audio,
                "sampling_rate": sampling_rate,
            },
            **call_kwargs,
        )

    return result["text"].strip()


def transcribe(loaded_model: dict, audio: "np.ndarray") -> str:
    backend = loaded_model.get("backend", "ctc")

    if backend == "ctc":
        return _transcribe_ctc(loaded_model, audio)

    if backend == "whisper":
        return _transcribe_whisper(loaded_model, audio)

    raise ValueError(f"Unsupported backend: {backend}")
