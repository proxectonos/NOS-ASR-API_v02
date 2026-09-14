#!flask/bin/python
import os
import time

from flask import Flask, render_template, request, jsonify

from utils.audio import load_audio
from utils.model_loader import read_config, load_models, transcribe


MODELS_ROOT = os.getenv("MODELS_ROOT", "models")
CONFIG_JSON_PATH = os.getenv("ASR_API_CONFIG", "config.json")
if not os.path.exists(CONFIG_JSON_PATH) and os.path.exists("config.example.json"):
    CONFIG_JSON_PATH = "config.example.json"
USE_CUDA = os.getenv("USE_CUDA") == "1"
MAX_AUDIO_MB = int(os.getenv("MAX_AUDIO_MB", "50"))


config_data = read_config(CONFIG_JSON_PATH)
loaded_models, default_model_ids = load_models(config_data, MODELS_ROOT, USE_CUDA)

print("USE_CUDA", USE_CUDA)
print("LOADED MODELS:", list(loaded_models.keys()))
print("DEFAULTS:", default_model_ids)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_AUDIO_MB * 1024 * 1024

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({
        "status": "ok",
        "loaded_models": list(loaded_models.keys()),
        "defaults": default_model_ids,
        "use_cuda": USE_CUDA,
    }), 200

def _public_model_options(model: dict) -> dict:
    """
    Return only public, non-sensitive model options.

    This is useful for debugging the API from /api/asr/models without exposing
    arbitrary internal configuration values.
    """
    options = model.get("options", {}) or {}

    public_keys = {
        "chunk_length_s",
        "stride_length_s",
        "batch_size",
        "max_new_tokens",
        "language",
        "task",
        "num_beams",
        "return_timestamps",
        "remove_inverted_question_marks",
        "remove_inverted_exclamation_marks",
    }

    return {
        key: value
        for key, value in options.items()
        if key in public_keys
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        models={k: loaded_models[k]["language"] for k in loaded_models.keys()},
    )


@app.route("/api/asr/models", methods=["GET"])
def list_models():
    by_lang = {}

    for mid, m in loaded_models.items():
        lang = m["lang"]

        if lang not in by_lang:
            by_lang[lang] = {
                "name": m["language"],
                "models": {},
            }

        by_lang[lang]["models"][mid] = {
            "default": default_model_ids.get(lang) == mid,
            "model_type": m.get("model_type"),
            "backend": m.get("backend"),
            "sampling_rate": m.get("sampling_rate"),
            "has_lm": m.get("has_lm", False),
            "hf_repo": m.get("hf_repo"),
            "options": _public_model_options(m),
        }

    return jsonify(by_lang), 200


@app.route("/api/asr/check", methods=["GET"])
def check(model_id=None, lang=None):
    if not model_id and not lang:
        model_id = request.args.get("model_id")
        lang = request.args.get("lang")

    if model_id:
        if model_id not in loaded_models:
            return jsonify({"message": f"Model {model_id} not found"}), 400
        if lang and loaded_models[model_id]["lang"] != lang:
            return jsonify({"message": f"Model {model_id} not in lang {lang}"}), 400

    elif lang:
        if lang in default_model_ids:
            model_id = default_model_ids[lang]
        else:
            return jsonify({"message": f"No model for language {lang}"}), 400

    else:
        return jsonify({"message": "Request must specify model_id or lang"}), 400

    model = loaded_models[model_id]

    return {
        "model_id": model_id,
        "lang": model["lang"],
        "model_type": model.get("model_type"),
        "backend": model.get("backend"),
        "sampling_rate": model["sampling_rate"],
    }, 200


@app.route("/api/asr", methods=["POST"])
def asr():
    model_id = request.args.get("model_id") or request.form.get("model_id")
    lang = request.args.get("lang") or request.form.get("lang")

    if "audio" not in request.files:
        return jsonify({"message": "Missing 'audio' file field"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    if not audio_bytes:
        return jsonify({"message": "Empty audio file"}), 400

    r, status = check(model_id, lang)

    if status != 200:
        return r, status

    model_id = r["model_id"]
    sr = r["sampling_rate"]
    model = loaded_models[model_id]

    print(
        f"ASR API REQUEST: "
        f"model={model_id} "
        f"type={model.get('model_type')} "
        f"bytes={len(audio_bytes)} "
        f"filename={audio_file.filename}"
    )

    try:
        audio = load_audio(
            audio_bytes,
            target_sr=sr,
            filename=audio_file.filename,
        )
    except Exception as e:
        return jsonify({"message": f"Failed to decode audio: {e}"}), 400

    if audio.size == 0:
        return jsonify({"message": "Decoded audio is empty"}), 400

    duration_s = float(len(audio)) / float(sr)

    try:
        t0 = time.time()
        text = transcribe(model, audio)
        inference_s = time.time() - t0
    except Exception as e:
        print(f"INFERENCE ERROR: {e}")
        return jsonify({"message": f"Inference failed: {e}"}), 500

    return jsonify({
        "model_id": model_id,
        "model_type": model.get("model_type"),
        "backend": model.get("backend"),
        "lang": model["lang"],
        "text": text,
        "duration_s": duration_s,
        "inference_s": inference_s,
    }), 200


def main():
    app.run(debug=True, host="::", port=5002)


if __name__ == "__main__":
    main()
