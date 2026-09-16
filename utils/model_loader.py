import json
import os
import threading
from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers import (
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)


# One inference at a time per process when using CUDA.
# This avoids concurrent requests duplicating temporary tensors on the GPU.
_CUDA_INFERENCE_LOCK = threading.Lock()


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
    options: Optional[Dict[str, Any]] = None,
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


def _log_tokenizer_diagnostics(model_id: str, tokenizer: Any) -> None:
    print(f"[loader] {model_id} tokenizer class: {tokenizer.__class__.__name__}")
    print(
        f"[loader] {model_id} word_delimiter_token: "
        f"{getattr(tokenizer, 'word_delimiter_token', None)!r}"
    )
    print(f"[loader] {model_id} unk_token: {getattr(tokenizer, 'unk_token', None)!r}")
    print(f"[loader] {model_id} pad_token: {getattr(tokenizer, 'pad_token', None)!r}")

    try:
        vocab = tokenizer.get_vocab()
    except Exception as e:
        print(f"[loader] {model_id} tokenizer vocab diagnostics unavailable: {e}")
        return

    selected = {}
    for token in [
        " ",
        "|",
        "[UNK]",
        "[PAD]",
        "l",
        "r",
        "c",
        "g",
        "u",
        "e",
        "i",
        "ll",
        "rr",
        "cc",
        "gue",
        "gui",
    ]:
        if token in vocab:
            selected[token] = vocab[token]

    print(f"[loader] {model_id} tokenizer vocab size: {len(vocab)}")
    print(f"[loader] {model_id} selected vocab entries: {selected}")


def _store_processor_parts(record: dict, processor: Any) -> None:
    """
    Store tokenizer and feature extractor explicitly when available.

    This makes CTC pipeline emulation independent from processor.batch_decode().
    """
    if hasattr(processor, "tokenizer"):
        record["tokenizer"] = processor.tokenizer

    if hasattr(processor, "feature_extractor"):
        record["feature_extractor"] = processor.feature_extractor


def _load_wav2vec2_lm(
    *,
    entry: dict,
    config_data: dict,
    cache_dir: str,
    device: torch.device,
) -> dict:
    """
    Load Wav2Vec2 with LM.

    This backend intentionally keeps the existing LM decoder path. Do not route
    Wav2Vec2+LM through the emulated plain-CTC pipeline.
    """
    model_id = entry["model_id"]
    lang = entry["lang"]
    hf_repo = entry["hf_repo"]
    sampling_rate = entry.get("sampling_rate", 16000)
    model_type = entry.get("model_type", "wav2vec2_lm")
    options = entry.get("options", {})

    try:
        processor = Wav2Vec2ProcessorWithLM.from_pretrained(
            hf_repo,
            cache_dir=cache_dir,
        )
        has_lm = True
    except Exception as e:
        print(
            f"[loader] LM processor failed for {model_id} ({e}); "
            "falling back to plain Wav2Vec2Processor"
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
        options=options,
    )
    record["processor"] = processor
    record["model"] = model
    _store_processor_parts(record, processor)
    return record


def _load_wav2vec2(
    *,
    entry: dict,
    config_data: dict,
    cache_dir: str,
    device: torch.device,
) -> dict:
    """
    Load plain Wav2Vec2 CTC.

    If options.word_delimiter_token is provided, build the tokenizer explicitly.
    This is useful for the Galician XLS-R 300M model, where the production/demo
    setup uses word_delimiter_token=" " rather than the added "|" token.

    Inference for this backend uses the emulated HF CTC pipeline path.
    """
    model_id = entry["model_id"]
    lang = entry["lang"]
    hf_repo = entry["hf_repo"]
    sampling_rate = entry.get("sampling_rate", 16000)
    model_type = entry.get("model_type", "wav2vec2")
    options = entry.get("options", {}) or {}

    word_delimiter_token = options.get("word_delimiter_token")

    if word_delimiter_token is not None:
        print(
            f"[loader] loading {model_id} Wav2Vec2 tokenizer explicitly "
            f"with word_delimiter_token={word_delimiter_token!r}"
        )
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            hf_repo,
            cache_dir=cache_dir,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token=word_delimiter_token,
        )

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            hf_repo,
            cache_dir=cache_dir,
        )

        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )

        # Keep the demo behaviour when hf_repo is a local writable directory.
        # If hf_repo is a Hub id or not writable, this is not required and should
        # not block loading.
        try:
            if os.path.isdir(hf_repo) and os.access(hf_repo, os.W_OK):
                tokenizer.save_pretrained(hf_repo)
                print(f"[loader] saved explicit tokenizer into {hf_repo}")
        except Exception as e:
            print(f"[loader] could not save explicit tokenizer into {hf_repo}: {e}")

    else:
        processor = Wav2Vec2Processor.from_pretrained(
            hf_repo,
            cache_dir=cache_dir,
        )

    model = Wav2Vec2ForCTC.from_pretrained(
        hf_repo,
        cache_dir=cache_dir,
    ).to(device).eval()

    record = _base_model_record(
        backend="ctc_pipeline_emulated",
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
    record["processor"] = processor
    record["model"] = model
    _store_processor_parts(record, processor)

    if "tokenizer" in record:
        _log_tokenizer_diagnostics(model_id, record["tokenizer"])

    print(
        f"[loader] {model_id} backend=ctc_pipeline_emulated "
        f"inputs_to_logits_ratio={_model_align_to(model)} "
        f"model_vocab_size={getattr(model.config, 'vocab_size', None)}"
    )

    return record


def _load_w2v_bert(
    *,
    entry: dict,
    config_data: dict,
    cache_dir: str,
    device: torch.device,
) -> dict:
    """
    Load W2V-BERT using only the complete processor available with the model.

    No fallback processor is built here. If the model package does not contain a
    complete Wav2Vec2BertProcessor, loading fails intentionally instead of
    silently using a different tokenizer/feature-extractor combination.

    Inference for this backend uses the emulated HF CTC pipeline path.
    """
    model_id = entry["model_id"]
    lang = entry["lang"]
    hf_repo = entry["hf_repo"]
    sampling_rate = entry.get("sampling_rate", 16000)
    model_type = entry.get("model_type", "w2v_bert")
    options = entry.get("options", {}) or {}

    processor = Wav2Vec2BertProcessor.from_pretrained(
        hf_repo,
        cache_dir=cache_dir,
    )

    model = Wav2Vec2BertForCTC.from_pretrained(
        hf_repo,
        cache_dir=cache_dir,
    ).to(device).eval()

    record = _base_model_record(
        backend="ctc_pipeline_emulated",
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
    record["processor"] = processor
    record["model"] = model
    _store_processor_parts(record, processor)

    if "tokenizer" in record:
        _log_tokenizer_diagnostics(model_id, record["tokenizer"])

    print(
        f"[loader] {model_id} backend=ctc_pipeline_emulated "
        f"inputs_to_logits_ratio={_model_align_to(model)} "
        f"model_vocab_size={getattr(model.config, 'vocab_size', None)}"
    )

    return record


def _whisper_use_fp16(device: torch.device, options: Dict[str, Any]) -> bool:
    """
    Use fp16 on CUDA by default, except when stride-aware HF pipeline chunking
    is enabled. Some Transformers versions keep Whisper pipeline input_features
    as float32; using float32 for pipeline mode avoids dtype mismatches.
    """
    if device.type != "cuda":
        return False

    stride_length_s = options.get("stride_length_s")

    try:
        use_stride_pipeline = (
            stride_length_s is not None and float(stride_length_s) > 0
        )
    except Exception:
        use_stride_pipeline = False

    default_use_fp16 = not use_stride_pipeline
    return bool(options.get("use_fp16", default_use_fp16))


def _load_whisper(
    *,
    entry: dict,
    config_data: dict,
    cache_dir: str,
    device: torch.device,
) -> dict:
    model_id = entry["model_id"]
    lang = entry["lang"]
    hf_repo = entry["hf_repo"]
    sampling_rate = entry.get("sampling_rate", 16000)
    model_type = entry.get("model_type", "whisper")
    options = entry.get("options", {}) or {}

    language = options.get("language", "galician")
    task = options.get("task", "transcribe")

    processor = WhisperProcessor.from_pretrained(
        hf_repo,
        cache_dir=cache_dir,
        language=language,
        task=task,
    )

    use_fp16 = _whisper_use_fp16(device, options)
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            hf_repo,
            cache_dir=cache_dir,
            dtype=torch_dtype,
        ).to(device).eval()
    except TypeError:
        model = WhisperForConditionalGeneration.from_pretrained(
            hf_repo,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
        ).to(device).eval()

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
    record["processor"] = processor
    record["model"] = model

    # Used only when stride_length_s > 0.
    # Note: in environments with broken TorchCodec, Whisper pipeline chunking may
    # fail if the installed Transformers version touches TorchCodec internally.
    record["pipeline"] = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if device.type == "cuda" else -1,
        torch_dtype=torch_dtype,
    )

    return record


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
        model_type = entry.get("model_type", "wav2vec2_lm")
        cache_dir = _resolve_cache_dir(models_root, model_id)

        print(
            f"[loader] loading {model_id} "
            f"({model_type}) from {hf_repo} -> {cache_dir}"
        )

        if model_type == "wav2vec2_lm":
            record = _load_wav2vec2_lm(
                entry=entry,
                config_data=config_data,
                cache_dir=cache_dir,
                device=device,
            )

        elif model_type == "wav2vec2":
            record = _load_wav2vec2(
                entry=entry,
                config_data=config_data,
                cache_dir=cache_dir,
                device=device,
            )

        elif model_type in {"w2v_bert", "wav2vec2_bert"}:
            record = _load_w2v_bert(
                entry=entry,
                config_data=config_data,
                cache_dir=cache_dir,
                device=device,
            )

        elif model_type == "whisper":
            record = _load_whisper(
                entry=entry,
                config_data=config_data,
                cache_dir=cache_dir,
                device=device,
            )

        else:
            raise ValueError(
                f"Unsupported model_type '{model_type}' for model '{model_id}'"
            )

        loaded[model_id] = record

        if entry.get("default") or lang not in defaults_by_lang:
            defaults_by_lang[lang] = model_id

    return loaded, defaults_by_lang


# =============================================================================
# Shared tensor and CTC helpers
# =============================================================================

def _move_inputs_to_device(inputs, device: torch.device) -> dict:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _model_align_to(model) -> int:
    return int(getattr(model.config, "inputs_to_logits_ratio", 1) or 1)


def _rescale_stride_one(stride, ratio: float):
    """
    Convert an audio-sample stride tuple into token/logit-frame units.

    This mirrors the relevant stride rescaling used by the HF ASR pipeline for
    plain CTC models.
    """
    input_n, left, right = stride
    token_n = int(round(input_n * ratio))
    left_n = int(round(left / input_n * token_n)) if input_n else 0
    right_n = int(round(right / input_n * token_n)) if input_n else 0
    return token_n, left_n, right_n


def _ctc_get_tokenizer(loaded_model: dict):
    if loaded_model.get("tokenizer") is not None:
        return loaded_model["tokenizer"]

    processor = loaded_model.get("processor")
    if processor is not None and hasattr(processor, "tokenizer"):
        return processor.tokenizer

    raise RuntimeError("CTC tokenizer not available in loaded model record.")


def _ctc_get_feature_extractor(loaded_model: dict):
    if loaded_model.get("feature_extractor") is not None:
        return loaded_model["feature_extractor"]

    processor = loaded_model.get("processor")
    if processor is not None and hasattr(processor, "feature_extractor"):
        return processor.feature_extractor

    if processor is not None:
        return processor

    raise RuntimeError("CTC feature extractor not available in loaded model record.")


def _ctc_forward_token_ids(loaded_model: dict, audio: np.ndarray) -> torch.Tensor:
    """
    Forward one CTC chunk and return greedy token IDs.

    This avoids processor.batch_decode() during chunk recombination. The final
    decode happens once after overlap has been cropped in token-frame space.
    """
    feature_extractor = _ctc_get_feature_extractor(loaded_model)
    model = loaded_model["model"]
    device = loaded_model["device"]
    sampling_rate = loaded_model["sampling_rate"]

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    try:
        inputs = feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
    except TypeError:
        inputs = feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

    inputs = _move_inputs_to_device(inputs, device)

    with torch.inference_mode():
        token_ids = model(**inputs).logits.argmax(dim=-1).detach().cpu()

    del inputs

    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.empty_cache()

    return token_ids


def _decode_ctc_logits_with_lm(processor, logits: torch.Tensor) -> str:
    decoded = processor.batch_decode(logits.cpu().numpy())

    if hasattr(decoded, "text"):
        return decoded.text[0].strip()

    if isinstance(decoded, dict) and "text" in decoded:
        return decoded["text"][0].strip()

    return str(decoded[0]).strip()


def _decode_ctc_greedy(processor, logits: torch.Tensor) -> str:
    pred_ids = torch.argmax(logits, dim=-1)

    try:
        texts = processor.batch_decode(
            pred_ids.cpu(),
            skip_special_tokens=True,
        )
    except TypeError:
        texts = processor.batch_decode(pred_ids.cpu())

    return texts[0].strip()


def _transcribe_ctc_single(loaded_model: dict, audio: np.ndarray) -> str:
    """
    Legacy single-shot CTC path.

    This remains necessary for Wav2Vec2+LM and for any future CTC backend that
    should not use the plain CTC pipeline emulation.
    """
    processor = loaded_model["processor"]
    model = loaded_model["model"]
    device = loaded_model["device"]
    sampling_rate = loaded_model["sampling_rate"]

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    inputs = processor(
        audio,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )

    inputs = _move_inputs_to_device(inputs, device)

    with torch.inference_mode():
        # Important: do not hard-code input_values here.
        # Wav2Vec2 normally uses input_values, while W2V-BERT uses input_features.
        logits = model(**inputs).logits

    if loaded_model.get("has_lm", False):
        text = _decode_ctc_logits_with_lm(processor, logits)
    else:
        text = _decode_ctc_greedy(processor, logits)

    del inputs, logits

    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.empty_cache()

    return text.strip()


def _default_ctc_chunk_length_s(loaded_model: dict) -> Optional[float]:
    model_type = loaded_model.get("model_type")

    if model_type in {"w2v_bert", "wav2vec2_bert"}:
        return 30.0

    return None


def _transcribe_ctc_chunked(
    loaded_model: dict,
    audio: np.ndarray,
    chunk_length_s: float,
) -> str:
    """
    Legacy text-level chunking.

    This is kept for Wav2Vec2+LM fallback scenarios. Plain wav2vec2 and
    W2V-BERT should use _transcribe_ctc_pipeline_emulated() instead.
    """
    sampling_rate = loaded_model["sampling_rate"]
    model_id = loaded_model["model_id"]

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    chunk_size = int(float(chunk_length_s) * sampling_rate)

    if chunk_size <= 0:
        raise ValueError(f"Invalid chunk_length_s for {model_id}: {chunk_length_s}")

    texts = []
    total_chunks = int(np.ceil(len(audio) / float(chunk_size)))

    print(
        f"[transcribe] chunking {model_id}: "
        f"{len(audio) / float(sampling_rate):.2f}s audio, "
        f"{float(chunk_length_s):.2f}s chunks, "
        f"{total_chunks} chunks total"
    )

    for idx, start in enumerate(range(0, len(audio), chunk_size), start=1):
        end = min(start + chunk_size, len(audio))
        chunk = audio[start:end]

        if len(chunk) == 0:
            continue

        print(
            f"[transcribe] {model_id}: chunk {idx}/{total_chunks} "
            f"({len(chunk) / float(sampling_rate):.2f}s)"
        )

        text = _transcribe_ctc_single(loaded_model, chunk)

        if text:
            texts.append(text)

        if loaded_model["device"].type == "cuda":
            torch.cuda.empty_cache()

    return " ".join(texts).strip()


def _transcribe_ctc_pipeline_emulated(loaded_model: dict, audio: np.ndarray) -> str:
    """
    Emulate Hugging Face AutomaticSpeechRecognitionPipeline for plain CTC models.

    """
    model = loaded_model["model"]
    tokenizer = _ctc_get_tokenizer(loaded_model)
    sampling_rate = int(loaded_model.get("sampling_rate", 16000))
    options = loaded_model.get("options", {}) or {}

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    chunk_length_s = float(options.get("chunk_length_s", 30.0))
    stride_left_s = float(options.get("stride_left_s", 6.0))
    stride_right_s = float(options.get("stride_right_s", 3.0))
    skip_special_tokens = bool(options.get("decode_skip_special_tokens", False))

    align_to = _model_align_to(model)

    chunk_len = int(round(chunk_length_s * sampling_rate / align_to) * align_to)
    stride_left = int(round(stride_left_s * sampling_rate / align_to) * align_to)
    stride_right = int(round(stride_right_s * sampling_rate / align_to) * align_to)

    if chunk_len <= 0:
        raise ValueError(f"Invalid chunk_length_s: {chunk_length_s}")

    if chunk_len < stride_left + stride_right:
        raise ValueError(
            "Chunk length must be greater than stride length: "
            f"chunk={chunk_length_s}, stride_left={stride_left_s}, "
            f"stride_right={stride_right_s}"
        )

    step = chunk_len - stride_left - stride_right
    inputs_len = audio.shape[0]

    print(
        f"[transcribe] Emulated HF CTC pipeline {loaded_model.get('model_id')}: "
        f"{inputs_len / float(sampling_rate):.2f}s audio, "
        f"{chunk_length_s:.2f}s chunks, "
        f"stride=({stride_left_s:.2f}s,{stride_right_s:.2f}s), "
        f"align_to={align_to}, "
        f"decode_skip_special_tokens={skip_special_tokens}"
    )

    final_items = []
    total_chunks = 0

    for chunk_start_idx in range(0, inputs_len, step):
        chunk_end_idx = chunk_start_idx + chunk_len
        chunk = audio[chunk_start_idx:chunk_end_idx]

        if chunk.shape[0] == 0:
            continue

        is_first = chunk_start_idx == 0
        is_last = chunk_end_idx >= inputs_len

        current_stride_left = 0 if is_first else stride_left
        current_stride_right = 0 if is_last else stride_right

        total_chunks += 1

        print(
            f"[transcribe] {loaded_model.get('model_id')}: "
            f"emulated chunk {total_chunks} "
            f"start={chunk_start_idx / float(sampling_rate):.2f}s "
            f"end={min(chunk_end_idx, inputs_len) / float(sampling_rate):.2f}s "
            f"chunk={chunk.shape[0] / float(sampling_rate):.2f}s"
        )

        token_ids = _ctc_forward_token_ids(loaded_model, chunk)

        ratio = 1.0 / float(align_to)
        total_n, left_n, right_n = _rescale_stride_one(
            (
                int(chunk.shape[0]),
                int(current_stride_left),
                int(current_stride_right),
            ),
            ratio,
        )

        items = token_ids.numpy()
        right_boundary = max(left_n, total_n - right_n)
        cropped = items[:, left_n:right_boundary]

        if cropped.shape[1] > 0:
            final_items.append(cropped)

        del token_ids

        if is_last:
            break

    if not final_items:
        return ""

    merged = np.concatenate(final_items, axis=1).squeeze(0)

    text = tokenizer.decode(
        merged,
        skip_special_tokens=skip_special_tokens,
    )

    print(
        f"[transcribe] Emulated HF CTC pipeline decoded "
        f"{len(final_items)} chunks into {merged.shape[0]} token frames."
    )

    return text.strip()


def _transcribe_ctc(loaded_model: dict, audio: np.ndarray) -> str:
    sampling_rate = loaded_model["sampling_rate"]
    options = loaded_model.get("options", {}) or {}
    backend = loaded_model.get("backend", "ctc")

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration_s = len(audio) / float(sampling_rate)

    # Do not use plain CTC emulation for LM decoding.
    if not loaded_model.get("has_lm", False) and backend == "ctc_pipeline_emulated":
        return _transcribe_ctc_pipeline_emulated(loaded_model, audio)

    chunk_length_s = options.get(
        "chunk_length_s",
        _default_ctc_chunk_length_s(loaded_model),
    )

    if chunk_length_s is not None and duration_s > float(chunk_length_s):
        return _transcribe_ctc_chunked(
            loaded_model,
            audio,
            chunk_length_s=float(chunk_length_s),
        )

    return _transcribe_ctc_single(loaded_model, audio)


# =============================================================================
# Whisper helpers
# =============================================================================

def _whisper_generate_kwargs(loaded_model: dict) -> dict:
    processor = loaded_model["processor"]
    options = loaded_model.get("options", {}) or {}

    generate_kwargs = {}

    if "max_new_tokens" in options:
        generate_kwargs["max_new_tokens"] = int(options["max_new_tokens"])

    if "num_beams" in options:
        generate_kwargs["num_beams"] = int(options["num_beams"])

    language = options.get("language", "galician")
    task = options.get("task", "transcribe")

    # Used for the direct Whisper generate() path.
    if hasattr(processor, "get_decoder_prompt_ids"):
        try:
            generate_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
                language=language,
                task=task,
            )
        except Exception as e:
            print(
                f"[whisper] could not set forced_decoder_ids "
                f"for language={language}, task={task}: {e}"
            )

    return generate_kwargs


def _whisper_pipeline_generate_kwargs(loaded_model: dict) -> dict:
    options = loaded_model.get("options", {}) or {}

    generate_kwargs = {}

    if "max_new_tokens" in options:
        generate_kwargs["max_new_tokens"] = int(options["max_new_tokens"])

    if "num_beams" in options:
        generate_kwargs["num_beams"] = int(options["num_beams"])

    # For the HF ASR pipeline, pass language/task directly.
    if "language" in options:
        generate_kwargs["language"] = options["language"]

    if "task" in options:
        generate_kwargs["task"] = options["task"]

    return generate_kwargs


def _transcribe_whisper_single(loaded_model: dict, audio: np.ndarray) -> str:
    processor = loaded_model["processor"]
    model = loaded_model["model"]
    device = loaded_model["device"]
    sampling_rate = loaded_model["sampling_rate"]

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    inputs = processor(
        audio,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )

    # If Whisper is loaded in fp16 on CUDA, input_features must also be fp16.
    model_dtype = next(model.parameters()).dtype

    input_features = inputs["input_features"].to(
        device=device,
        dtype=model_dtype,
    )

    generate_kwargs = _whisper_generate_kwargs(loaded_model)

    with torch.inference_mode():
        pred_ids = model.generate(
            input_features,
            **generate_kwargs,
        )

    texts = processor.batch_decode(
        pred_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    del inputs, input_features, pred_ids

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return texts[0].strip()


def _transcribe_whisper_chunked(
    loaded_model: dict,
    audio: np.ndarray,
    chunk_length_s: float,
) -> str:
    """
    Simple non-overlapping Whisper chunking.

    This is the safer fallback for long audio when stride_length_s is not set.
    It avoids OOM, but may produce minor boundary artifacts.
    """
    sampling_rate = loaded_model["sampling_rate"]
    model_id = loaded_model["model_id"]

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    chunk_size = int(float(chunk_length_s) * sampling_rate)

    if chunk_size <= 0:
        raise ValueError(f"Invalid chunk_length_s for {model_id}: {chunk_length_s}")

    texts = []
    total_chunks = int(np.ceil(len(audio) / float(chunk_size)))

    print(
        f"[transcribe] chunking {model_id}: "
        f"{len(audio) / float(sampling_rate):.2f}s audio, "
        f"{float(chunk_length_s):.2f}s chunks, "
        f"{total_chunks} chunks total"
    )

    for idx, start in enumerate(range(0, len(audio), chunk_size), start=1):
        end = min(start + chunk_size, len(audio))
        chunk = audio[start:end]

        if len(chunk) == 0:
            continue

        print(
            f"[transcribe] {model_id}: chunk {idx}/{total_chunks} "
            f"({len(chunk) / float(sampling_rate):.2f}s)"
        )

        text = _transcribe_whisper_single(loaded_model, chunk)

        if text:
            texts.append(text)

        if loaded_model["device"].type == "cuda":
            torch.cuda.empty_cache()

    return " ".join(texts).strip()


def _transcribe_whisper_pipeline_chunked(
    loaded_model: dict,
    audio: np.ndarray,
    chunk_length_s: float,
    stride_length_s: float,
) -> str:
    """
    Stride-aware Whisper chunking through the Hugging Face ASR pipeline.

    Use this only when stride_length_s > 0. It gives the model acoustic context
    at chunk boundaries and lets the pipeline recombine the result.
    """
    pipe = loaded_model.get("pipeline")

    if pipe is None:
        return _transcribe_whisper_chunked(
            loaded_model,
            audio,
            chunk_length_s=chunk_length_s,
        )

    sampling_rate = loaded_model["sampling_rate"]
    model_id = loaded_model["model_id"]
    options = loaded_model.get("options", {}) or {}

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if chunk_length_s <= 0:
        raise ValueError(f"Invalid chunk_length_s for {model_id}: {chunk_length_s}")

    if stride_length_s <= 0:
        raise ValueError(f"Invalid stride_length_s for {model_id}: {stride_length_s}")

    if stride_length_s >= chunk_length_s:
        raise ValueError(
            f"stride_length_s must be smaller than chunk_length_s for {model_id}: "
            f"stride={stride_length_s}, chunk={chunk_length_s}"
        )

    generate_kwargs = _whisper_pipeline_generate_kwargs(loaded_model)

    return_timestamps = bool(options.get("return_timestamps", True))

    pipe_kwargs = {
        "chunk_length_s": float(chunk_length_s),
        "stride_length_s": float(stride_length_s),
        "generate_kwargs": generate_kwargs,
        "return_timestamps": return_timestamps,
        "decoder_kwargs": {
            "clean_up_tokenization_spaces": False,
        },
    }

    if "batch_size" in options:
        pipe_kwargs["batch_size"] = int(options["batch_size"])

    print(
        f"[transcribe] HF pipeline chunking {model_id}: "
        f"{len(audio) / float(sampling_rate):.2f}s audio, "
        f"{float(chunk_length_s):.2f}s chunks, "
        f"{float(stride_length_s):.2f}s stride, "
        f"return_timestamps={return_timestamps}"
    )

    result = pipe(
        {
            "array": audio,
            "sampling_rate": sampling_rate,
        },
        **pipe_kwargs,
    )

    if isinstance(result, dict):
        return str(result.get("text", "")).strip()

    return str(result).strip()


def _transcribe_whisper(loaded_model: dict, audio: np.ndarray) -> str:
    sampling_rate = loaded_model["sampling_rate"]
    options = loaded_model.get("options", {}) or {}

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration_s = len(audio) / float(sampling_rate)

    chunk_length_s = options.get("chunk_length_s")
    stride_length_s = options.get("stride_length_s")

    if chunk_length_s is not None and duration_s > float(chunk_length_s):
        if stride_length_s is not None and float(stride_length_s) > 0:
            return _transcribe_whisper_pipeline_chunked(
                loaded_model,
                audio,
                chunk_length_s=float(chunk_length_s),
                stride_length_s=float(stride_length_s),
            )

        return _transcribe_whisper_chunked(
            loaded_model,
            audio,
            chunk_length_s=float(chunk_length_s),
        )

    return _transcribe_whisper_single(loaded_model, audio)


# =============================================================================
# Public transcription dispatch
# =============================================================================

def _transcribe_unlocked(loaded_model: dict, audio: np.ndarray) -> str:
    backend = loaded_model.get("backend", "ctc")

    if backend in {"ctc", "ctc_pipeline_emulated"}:
        return _transcribe_ctc(loaded_model, audio)

    if backend == "whisper":
        return _transcribe_whisper(loaded_model, audio)

    raise ValueError(f"Unsupported backend: {backend}")


def _postprocess_text(loaded_model: dict, text: str) -> str:
    options = loaded_model.get("options", {}) or {}

    if options.get("remove_inverted_question_marks", False):
        text = text.replace("¿", "")

    if options.get("remove_inverted_exclamation_marks", False):
        text = text.replace("¡", "")

    return text.strip()


def transcribe(loaded_model: dict, audio: np.ndarray) -> str:
    device = loaded_model.get("device")

    if isinstance(device, torch.device) and device.type == "cuda":
        with _CUDA_INFERENCE_LOCK:
            text = _transcribe_unlocked(loaded_model, audio)
    else:
        text = _transcribe_unlocked(loaded_model, audio)

    return _postprocess_text(loaded_model, text)
