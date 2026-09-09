# NOS-ASR-API

Dockerized HTTP API for Galician Automatic Speech Recognition.

The API serves Galician ASR models from [Proxecto Nós](https://nos.gal/gl/proxecto-nos) through a small Flask + Gunicorn server. Models are configured in `config.json` and selected at request time with `model_id` or, when omitted, through the default model for a language.

The API currently supports four backend families:

- `wav2vec2_lm`: Wav2Vec2 CTC with a KenLM decoder.
- `wav2vec2`: Wav2Vec2 CTC with greedy decoding.
- `w2v_bert`: W2V-BERT CTC with greedy decoding.
- `whisper`: Whisper encoder-decoder ASR, with optional stride-aware long-audio chunking.

---

## Setup

Clone the repository and create the local model directory:

```bash
git clone <this-repo-url> NOS-ASR-API
cd NOS-ASR-API
mkdir -p models
```

Create your local configuration from the example file:

```bash
cp config.example.json config.json
```

Models are pulled automatically from Hugging Face the first time the server starts. The Hugging Face cache is stored under `models/`, or under the directory pointed to by `MODELS_ROOT`.

You can find the Proxecto Nós ASR models on Hugging Face:

<https://huggingface.co/collections/proxectonos/asr-models>

For private or gated repositories, set `HF_TOKEN` before starting the API:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

With Docker Compose, this can be passed through the `environment` section or through an `.env` file.

---

## Available model types

| `model_type` | Backend | Decoder | Typical use |
|---|---|---|---|
| `wav2vec2_lm` | `Wav2Vec2ForCTC` | KenLM-rescored CTC | More fluent output when RAM/disk budget is available |
| `wav2vec2` | `Wav2Vec2ForCTC` | Greedy CTC | Lighter CPU-friendly deployment and testing |
| `w2v_bert` | `Wav2Vec2BertForCTC` | Greedy CTC | W2V-BERT CTC checkpoints |
| `whisper` | `WhisperForConditionalGeneration` / Transformers ASR pipeline | Encoder-decoder generation | Whisper ASR models, especially on GPU |

---

## Proxecto Nós ASR models

The following entries are intended for the current API configuration.

| `model_id` | `hf_repo` | `model_type` | Notes |
|---|---|---|---|
| `nos_asr_gl` | `proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm` | `wav2vec2_lm` | Wav2Vec2 XLSR-53 large with bundled KenLM language model. Heavier, but LM-rescored. |
| `nos_asr_gl_300m` | `proxectonos/Nos_ASR-wav2vec2-xls-r-300m-gl` | `wav2vec2` | Wav2Vec2 XLS-R 300M, greedy CTC. Lighter and suitable for local CPU tests. |
| `w2v_bert_2_gl` | `proxectonos/w2v-bert-2.0-gl` | `w2v_bert` | W2V-BERT CTC|
| `whisper_large_v3_turbo_gl_v1` | `proxectonos/whisper-large-v3-turbo-gl-v1.0` | `whisper` | Whisper Large-v3-Turbo fine-tuned for Galician ASR. Recommended on GPU. |

---

## Configuration

The API reads its model registry from `config.json` by default. You can override the path with `ASR_API_CONFIG`.

Only models with `"load": true` are loaded at server startup and can be selected by `model_id`. Models with `"load": false` are documented in the registry but are not available until they are activated and the server is restarted.

The `default` flag controls which model is used when the request specifies only `?lang=gl`.

### Example configuration

```json
{
  "languages": {
    "gl": "galego"
  },
  "models": [
    {
      "model_id": "nos_asr_gl_300m",
      "lang": "gl",
      "model_type": "wav2vec2",
      "hf_repo": "proxectonos/Nos_ASR-wav2vec2-xls-r-300m-gl",
      "sampling_rate": 16000,
      "load": true,
      "default": true
    },
    {
      "model_id": "nos_asr_gl",
      "lang": "gl",
      "model_type": "wav2vec2_lm",
      "hf_repo": "proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm",
      "sampling_rate": 16000,
      "load": false,
      "default": false
    },
    {
      "model_id": "w2v_bert_2_gl",
      "lang": "gl",
      "model_type": "w2v_bert",
      "hf_repo": "proxectonos/w2v-bert-2.0-gl",
      "sampling_rate": 16000,
      "load": false,
      "default": false,
      "options": {
        "chunk_length_s": 30
      }
    },
    {
      "model_id": "whisper_large_v3_turbo_gl_v1",
      "lang": "gl",
      "model_type": "whisper",
      "hf_repo": "proxectonos/whisper-large-v3-turbo-gl-v1.0",
      "sampling_rate": 16000,
      "load": false,
      "default": false,
      "options": {
        "language": "galician",
        "task": "transcribe",
        "chunk_length_s": 25,
        "stride_length_s": 5,
        "max_new_tokens": 256,
        "return_timestamps": true,
        "remove_inverted_question_marks": true,
        "remove_inverted_exclamation_marks": true
      }
    }
  ]
}
```

### Configuration fields

Top-level fields:

- `languages`: dictionary mapping language codes to language names.
- `models`: list of ASR model entries.

Model entry fields:

- `model_id`: public identifier used in API calls and as the local model cache subdirectory.
- `lang`: language code for the model. It must match a key in `languages`.
- `model_type`: backend identifier. Supported values are `wav2vec2_lm`, `wav2vec2`, `w2v_bert` and `whisper`.
- `hf_repo`: Hugging Face repository used to download the model and processor.
- `sampling_rate`: sampling rate expected by the model. Incoming audio is converted to this rate.
- `load`: set to `true` to load the model at API startup.
- `default`: set to `true` to use this model for requests that provide `?lang=<code>` without `model_id`.
- `options`: optional backend-specific options.

Common `options`:

- `chunk_length_s`: split long audio into chunks of this duration before inference.

Whisper-specific `options`:

- `language`: Whisper generation language. For the current Galician Whisper model, use `"galician"`.
- `task`: Whisper task, usually `"transcribe"`.
- `chunk_length_s`: chunk length for long audio.
- `stride_length_s`: overlap/context between chunks when using the Hugging Face ASR pipeline.
- `max_new_tokens`: maximum number of generated tokens per chunk.
- `num_beams`: optional number of beams for generation.
- `return_timestamps`: request timestamps during long-form Whisper inference. Recommended when using `stride_length_s`.
- `batch_size`: optional ASR pipeline batch size.
- `remove_inverted_question_marks`: remove `¿` from the final transcription.
- `remove_inverted_exclamation_marks`: remove `¡` from the final transcription.
- `use_fp16`: optional. When omitted, Whisper uses fp16 on CUDA for direct inference, but uses fp32 by default for stride-aware pipeline inference to avoid dtype mismatches in some Transformers versions.

---

## Run with Docker Compose

Docker Compose is the recommended deployment option. It installs the Python dependencies, `ffmpeg`, `libsndfile` and the native build toolchain required by `kenlm`.

Build and start:

```bash
docker compose build
docker compose up -d
```

Follow logs:

```bash
docker compose logs -f
```

Stop the service:

```bash
docker compose down
```

When only `config.json` changes, rebuilding is not necessary if the file is mounted into the container by `docker-compose.yml`; restarting the service is enough:

```bash
docker compose restart asr
```

The default Docker Compose configuration should use CPU:

```yaml
environment:
  - ASR_API_CONFIG=/app/config.json
  - MODELS_ROOT=/app/models
  - USE_CUDA=0
  - HF_HOME=/app/models/.hf_cache
  - MAX_AUDIO_MB=500
```

For GPU inference, set `USE_CUDA=1` and expose an NVIDIA GPU to the container:

```yaml
environment:
  - USE_CUDA=1

deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

GPU inference requires an NVIDIA GPU, CUDA-compatible PyTorch and the NVIDIA Container Toolkit.

Keep `--workers 1` for GPU deployments: each worker loads a full copy of each active model. Scale with multiple containers or replicas instead of multiple workers per container.

---

## Run locally

Native installs need:

- Python 3.10 or newer.
- `ffmpeg` available on `PATH`.
- A C++ build toolchain for `kenlm` when using `wav2vec2_lm`.
- The Python dependencies in `requirements.txt`.

System dependencies:

```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install -y build-essential cmake libboost-all-dev libeigen3-dev ffmpeg libsndfile1

# macOS
brew install cmake boost eigen ffmpeg libsndfile
```

Create and activate a Python environment:

```bash
python -m venv asr

# Linux/macOS
source asr/bin/activate

# Windows
.\asr\Scripts\activate

pip install -r requirements.txt
```

Run the API:

```bash
python run.py
```

On Linux/macOS you can also run Gunicorn manually:

```bash
gunicorn server:app -b :5051 --workers 1 --threads 1 --timeout 3600
```

On native Windows, use `python run.py` or Waitress instead of Gunicorn.

---

## API usage

The primary endpoint is:

```text
POST /api/asr
```

It accepts `multipart/form-data` with:

- `audio`: audio file to transcribe.
- `model_id`: optional query or form parameter selecting a loaded model.
- `lang`: optional query or form parameter selecting the default loaded model for a language.

Supported audio formats:

```text
wav, flac, ogg, oga, opus, mp3, m4a, mp4, aac, webm
```

Incoming audio is decoded to mono `float32` PCM and resampled to the model sampling rate, usually 16 kHz.

### Explicit model selection

```bash
curl -X POST \
  -F "audio=@sample.wav" \
  "http://localhost:5051/api/asr?model_id=nos_asr_gl_300m"
```

### Default model for a language

```bash
curl -X POST \
  -F "audio=@sample.wav" \
  "http://localhost:5051/api/asr?lang=gl"
```

If neither `model_id` nor `lang` is provided, the request is rejected with `400`.

### Typical response

```json
{
  "model_id": "nos_asr_gl_300m",
  "model_type": "wav2vec2",
  "backend": "ctc",
  "lang": "gl",
  "text": "ola mundo",
  "duration_s": 1.42,
  "inference_s": 0.51
}
```

---

## Auxiliary endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Minimal web UI for uploading audio and viewing the transcription |
| `GET` | `/healthz` | Basic health check |
| `GET` | `/api/asr/models` | List loaded models grouped by language |
| `GET` | `/api/asr/check` | Validate a `model_id` or `lang` selection |

Examples:

```bash
curl -s http://localhost:5051/healthz | python3 -m json.tool
curl -s http://localhost:5051/api/asr/models | python3 -m json.tool
curl -s "http://localhost:5051/api/asr/check?model_id=nos_asr_gl_300m" | python3 -m json.tool
```

Example `/healthz` response:

```json
{
  "status": "ok",
  "loaded_models": ["nos_asr_gl_300m"],
  "defaults": {
    "gl": "nos_asr_gl_300m"
  },
  "use_cuda": false
}
```

---

## Long audio

The API accepts one audio file per request and returns the transcription only when inference finishes. It does not stream partial results through the HTTP API or the web UI.

For CTC models such as Wav2Vec2 and W2V-BERT, long audio can be split internally by setting:

```json
"options": {
  "chunk_length_s": 30
}
```

This prevents very long recordings from being passed to the model in a single forward pass and reduces the risk of out-of-memory errors.

For Whisper models, long audio can be processed with stride-aware chunking:

```json
"options": {
  "language": "galician",
  "task": "transcribe",
  "chunk_length_s": 25,
  "stride_length_s": 5,
  "max_new_tokens": 256,
  "return_timestamps": true
}
```

This gives Whisper additional acoustic context at chunk boundaries and usually improves continuity, although it is slower than simple non-overlapping chunking.

Some warnings about missing ending timestamps may appear when chunks cut through speech. These warnings do not necessarily mean that transcription failed.

For production workflows with very long recordings, consider external segmentation, VAD-based segmentation, or a dedicated batch-processing endpoint.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ASR_API_CONFIG` | `config.json` | Path to the model registry |
| `MODELS_ROOT` | `models` | Root directory for model caches |
| `HF_HOME` | `models/.hf_cache` | Hugging Face cache directory |
| `HF_TOKEN` | unset | Hugging Face token for private or gated models |
| `USE_CUDA` | `0` | Set to `1` to enable CUDA inference |
| `MAX_AUDIO_MB` | `50` | Upload size limit in megabytes |
| `PORT` | `5052` for local launcher | Port used by `run.py`, if present |
| `WEB_WORKERS` | `1` | Gunicorn workers, if controlled by a launcher |
| `WEB_THREADS` | `4` | Gunicorn threads, if controlled by a launcher |
| `WEB_TIMEOUT` | `300` | Request timeout in seconds, if controlled by a launcher |

---

## Demo page

Once the server is running, a simple web interface is available at:

- Docker Compose: <http://localhost:5051>
- Local launcher: <http://localhost:5052>

The page allows uploading an audio file and viewing the transcription returned by the API.

To customize the page:

- Edit `static/css/style.css` for styling.
- Edit `templates/index.html` for layout.


---

## Docker build context

Keep model caches, audio files and generated chunks out of the Docker build context. A typical `.dockerignore` should include:

```dockerignore
models/
models/.hf_cache/
.hf_cache/

__pycache__/
*.pyc
*.pyo
*.egg-info/

.git/
.gitignore

.venv/
venv/
env/

*.wav
*.mp3
*.flac
*.ogg
*.oga
*.opus
*.m4a
*.mp4
*.aac
*.webm

samples/
chunks/
outputs/
tmp/
temp/

*.zip
*.tar
*.tar.gz
*.7z
```

The `models/` directory should be mounted as a volume:

```yaml
volumes:
  - ./models:/app/models
```

This keeps model weights outside the Docker image while making them available to the running container.

---

## Repository hygiene

Do not commit local model caches, downloaded weights, large audio files or local configuration.

A typical `.gitignore` should include:

```gitignore
config.json

models/*
!models/.gitkeep

samples/*
!samples/README.md

*.wav
*.mp3
*.flac
*.ogg
*.oga
*.opus
*.m4a
*.mp4
*.aac
*.webm

__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
env/

*.log
tmp/
temp/

.DS_Store
.vscode/
.idea/
```

---

## Troubleshooting

### The API starts but a model is not available

Only models with `"load": true` are loaded at startup. Check:

```bash
curl -s http://localhost:5051/api/asr/models | python3 -m json.tool
```

Activate the desired model in `config.json` and restart the service:

```bash
docker compose restart asr
```

### The API fails because `config.json` is missing

Create it from the example file:

```bash
cp config.example.json config.json
```

### Whisper is slow or fails on long files

On CPU, Whisper is slow and memory-intensive. Use short files for testing and GPU for practical long-form inference.

For long audio, use:

```json
"options": {
  "chunk_length_s": 25,
  "stride_length_s": 5,
  "max_new_tokens": 256,
  "return_timestamps": true
}
```

If the worker is killed with a message such as `SIGKILL` or `Perhaps out of memory?`, reduce the file length, use a smaller model, or move the workload to a GPU server.

### W2V-BERT runs out of memory on long audio

Use internal chunking:

```json
"options": {
  "chunk_length_s": 30
}
```

For very long recordings, external segmentation may still be preferable.

### Warning: Whisper did not predict an ending timestamp

This can happen when a chunk ends in the middle of speech. If the final transcription is complete and coherent, the warning is usually not blocking.

### Warning: `clean_up_tokenization_spaces=True` for WhisperTokenizer

Whisper uses a BPE tokenizer. The API disables tokenization-space cleanup in the direct decode path and may also disable it in the pipeline path depending on the installed Transformers version. The warning affects text post-processing only; it does not affect acoustic inference.

### Docker build sends several gigabytes of context

Update `.dockerignore` so `models/`, audio files, chunks and archive files are not included in the build context.

### Docker cannot connect to the daemon

Start Docker Desktop or the Docker service, then verify:

```bash
docker info
```

On Windows with WSL2, `wsl --shutdown` followed by reopening Docker Desktop often resolves a stale engine state.

---

## License

Apache License 2.0.

See `LICENSE`.

---

## Acknowledgements

This work is funded by the Ministerio para la Transformación Digital y de la Función Pública — Funded by EU – NextGenerationEU within the framework of the project Desarrollo de Modelos ALIA.

Esta publicación del proyecto Desarrollo de Modelos ALIA está financiada por el Ministerio para la Transformación Digital y de la Función Pública y por el Plan de Recuperación, Transformación y Resiliencia – Financiado por la Unión Europea – NextGenerationEU.

We would also like to thank [Dimensiona](https://www.dimensiona.com/gl/sobre-nos/) for the technical development of v01 of this API.
