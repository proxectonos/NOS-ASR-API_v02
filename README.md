# NOS-ASR-API

Dockerized HTTP API for Galician Automatic Speech Recognition.

The API serves Galician ASR models from [Proxecto Nós](https://nos.gal/gl/proxecto-nos) through a small Flask + Gunicorn server. Models are configured in `config.json` and selected at request time with `model_id` or, when omitted, through the default model for a language.

The API supports the following backend families:

- `wav2vec2_lm`: Wav2Vec2 CTC with a KenLM decoder.
- `wav2vec2`: Wav2Vec2 CTC with greedy decoding.
- `whisper`: Whisper encoder-decoder ASR through the Transformers ASR pipeline.
- `wav2vec2_bert`: W2V-BERT CTC through `AutoProcessor` and `AutoModelForCTC`.

## Setup

Start by cloning this repository and creating your `models` directory:

```bash
git clone <this-repo-url> NOS-ASR-API
cd NOS-ASR-API
mkdir -p models
```

Models are pulled automatically from Hugging Face the first time the server starts. The Hugging Face cache is rooted under `models/<model_id>/` or under the directory pointed to by `MODELS_ROOT`. Subsequent starts can reuse the local cache.

You can find the Proxecto Nós models on Hugging Face: <https://huggingface.co/collections/proxectonos/asr-models>

For private or gated repositories, set `HF_TOKEN` in the environment before starting the API.

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

On Docker Compose, this can be passed through the `environment` section or an `.env` file.

## Available model types

| `model_type` | Backend | Decoder | Typical use |
|---|---|---|---|
| `wav2vec2_lm` | `Wav2Vec2ForCTC` | KenLM-rescored CTC | More fluent output when RAM/disk budget is available |
| `wav2vec2` | `Wav2Vec2ForCTC` | Greedy CTC | Lighter CPU-friendly deployment and testing |
| `whisper` | Transformers `pipeline("automatic-speech-recognition")` | Encoder-decoder generation | Whisper ASR models, especially on GPU |
| `wav2vec2_bert` | `AutoModelForCTC` | Greedy CTC | W2V-BERT CTC checkpoints exported in standard Transformers format |

## Proxecto Nós ASR models

The following model entries are intended for the current API configuration.

| `model_id` | `hf_repo` | `model_type` | Notes |
|---|---|---|---|
| `nos_asr_gl` | `proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm` | `wav2vec2_lm` | Wav2Vec2 XLSR-53 large with bundled KenLM language model. Heavier, but LM-rescored. |
| `nos_asr_gl_300m` | `proxectonos/Nos_ASR-wav2vec2-xls-r-300m-gl` | `wav2vec2` | Wav2Vec2 XLS-R 300M, greedy CTC. Lighter and suitable for local CPU tests. |
| `whisper_large_v3_turbo_gl_v1` | `proxectonos/whisper-large-v3-turbo-gl-v1.0` | `whisper` | Whisper Large-v3-Turbo fine-tuned for Galician ASR. Recommended on GPU; short files can be tested on CPU. |
| `w2v_bert_2_gl` | `proxectonos/w2v-bert-2.0-gl` | `wav2vec2_bert` | W2V-BERT CTC. |


## Configuration

The API reads its model registry from `config.json` by default. You can override the path with `ASR_API_CONFIG`.

Only models with `"load": true` are loaded at server startup and can be selected by `model_id`. Models with `"load": false` are documented in the registry but are not available until they are activated and the server is restarted.

The `default` flag controls which model is used when the request specifies only `?lang=gl`.

### Example: all supported entries, one model loaded

This configuration declares the supported ASR entries but loads only the lighter Wav2Vec2 model at startup.

```json
{
  "languages": {
    "gl": "Galician"
  },
  "models": [
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
      "model_id": "nos_asr_gl_300m",
      "lang": "gl",
      "model_type": "wav2vec2",
      "hf_repo": "proxectonos/Nos_ASR-wav2vec2-xls-r-300m-gl",
      "sampling_rate": 16000,
      "load": true,
      "default": true
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
        "language": "gl",
        "task": "transcribe",
        "chunk_length_s": 15,
        "stride_length_s": 2,
        "batch_size": 1,
        "max_new_tokens": 96,
        "return_timestamps": false
      }
    },
    {
      "model_id": "w2v_bert_2_gl",
      "lang": "gl",
      "model_type": "wav2vec2_bert",
      "hf_repo": "proxectonos/w2v-bert-2.0-gl",
      "sampling_rate": 16000,
      "load": false,
      "default": false
    }
  ]
}
```

### Example: Whisper as the active model

```json
{
  "languages": {
    "gl": "Galician"
  },
  "models": [
    {
      "model_id": "whisper_large_v3_turbo_gl_v1",
      "lang": "gl",
      "model_type": "whisper",
      "hf_repo": "proxectonos/whisper-large-v3-turbo-gl-v1.0",
      "sampling_rate": 16000,
      "load": true,
      "default": true,
      "options": {
        "language": "gl",
        "task": "transcribe",
        "chunk_length_s": 15,
        "stride_length_s": 2,
        "batch_size": 1,
        "max_new_tokens": 96,
        "return_timestamps": false
      }
    }
  ]
}
```

### Example: Wav2Vec2 with KenLM as the default model

```json
{
  "languages": {
    "gl": "Galician"
  },
  "models": [
    {
      "model_id": "nos_asr_gl",
      "lang": "gl",
      "model_type": "wav2vec2_lm",
      "hf_repo": "proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm",
      "sampling_rate": 16000,
      "load": true,
      "default": true
    }
  ]
}
```

### Configuration fields

Top-level fields:

- `languages`: Dictionary mapping language codes to language names.
- `models`: List of ASR model entries.

Model entry fields:

- `model_id`: Public identifier used in API calls and as the local model cache subdirectory.
- `lang`: Language code for the model. It must match a key in `languages`.
- `model_type`: Backend identifier. Supported values are `wav2vec2_lm`, `wav2vec2`, `whisper` and `wav2vec2_bert`.
- `hf_repo`: Hugging Face repository used to download the model and processor.
- `sampling_rate`: Sampling rate expected by the model. Incoming audio is converted to this rate.
- `load`: Set to `true` to load the model at API startup.
- `default`: Optional. Set to `true` to use this model for requests that provide `?lang=<code>` without `model_id`.
- `options`: Optional backend-specific options. Currently used by the Whisper backend.

Whisper `options`:

- `language`: Generation language, usually `"gl"`.
- `task`: Whisper task, usually `"transcribe"`.
- `chunk_length_s`: Chunk length used by the Transformers ASR pipeline.
- `stride_length_s`: Overlap between chunks.
- `batch_size`: Pipeline batch size.
- `max_new_tokens`: Maximum generated tokens per chunk.
- `return_timestamps`: Whether the pipeline should request timestamps.

## Model selection

Use `model_id` to select an explicit model:

```bash
curl -X POST \
  -F "audio=@sample.wav" \
  "http://localhost:5051/api/asr?model_id=nos_asr_gl_300m"
```

Whisper example:

```bash
curl -X POST \
  -F "audio=@sample.wav" \
  "http://localhost:5051/api/asr?model_id=whisper_large_v3_turbo_gl_v1"
```

Use `lang` to select the default loaded model for a language:

```bash
curl -X POST \
  -F "audio=@sample.wav" \
  "http://localhost:5051/api/asr?lang=gl"
```

If neither `model_id` nor `lang` is provided, the request is rejected with `400`.

## Installation

The server can be run with Docker Compose or with a local Python environment.

### Run with Docker Compose

Docker Compose is the recommended option for deployment. It installs the Python dependencies, `ffmpeg` and the native build toolchain required by `kenlm`.

```bash
docker compose build
docker compose up
```

Run in the background:

```bash
docker compose up -d
```

Stop the service:

```bash
docker compose down
```

When only `config.json` changes, rebuilding is not necessary if the file is mounted into the container by `docker-compose.yml`; restarting the service is enough:

```bash
docker compose down
docker compose up -d
```

### Docker build context

Keep model caches, audio files and generated chunks out of the Docker build context. A typical `.dockerignore` should include:

```dockerignore
models/
models/.hf_cache/
.hf_cache/

__pycache__/
*.pyc

.git/
.venv/
venv/
asr/

*.wav
*.mp3
*.flac
*.ogg
*.m4a
*.webm
*.mp4

chunks/
outputs/
tmp/
temp/

*.zip
*.tar
*.tar.gz
*.7z
```

The `models/` directory should be mounted as a volume, for example:

```yaml
volumes:
  - ./models:/app/models
```

This keeps model weights outside the Docker image while making them available to the running container.

### Run with local installation

Native installs need:

- Python 3.10 or newer.
- `ffmpeg` available on `PATH`.
- A C++ build toolchain for `kenlm` when using `wav2vec2_lm`.
- The Python dependencies in `requirements.txt`.

System dependencies:

```bash
# macOS
brew install cmake boost eigen ffmpeg

# Debian/Ubuntu
sudo apt-get update
sudo apt-get install -y build-essential cmake libboost-all-dev libeigen3-dev ffmpeg libsndfile1
```

On macOS with Apple Silicon, if the `kenlm` build fails to find Boost:

```bash
export BOOST_ROOT="$(brew --prefix boost)"
```

#### Windows

The recommended local route on Windows is WSL2 with Ubuntu. Install WSL2, open an Ubuntu shell and follow the Debian/Ubuntu instructions above.

Native Windows builds of `kenlm` require Visual Studio build tools, CMake, ffmpeg and Boost. If you use Chocolatey and vcpkg:

```bat
choco install -y visualstudio2022-workload-vctools cmake ffmpeg

git clone https://github.com/microsoft/vcpkg %USERPROFILE%\vcpkg
%USERPROFILE%\vcpkg\bootstrap-vcpkg.bat
%USERPROFILE%\vcpkg\vcpkg install boost-system boost-thread boost-program-options eigen3 zlib
setx BOOST_ROOT "%USERPROFILE%\vcpkg\installed\x64-windows"
```

Open a fresh terminal after installing system dependencies.

#### Python environment

```bash
python -m venv asr

# Linux/macOS
source asr/bin/activate

# Windows
.\asr\Scripts\activate

pip install -r requirements.txt
```

Run through the cross-platform launcher:

```bash
python run.py
```

Convenience wrappers:

```bash
./run_local.sh       # Linux / macOS / WSL
run_local.bat        # Windows
```

The local launcher binds to port `5052` by default so it does not clash with Docker Compose on port `5051`.

### Manual Gunicorn launch

On Linux/macOS:

```bash
gunicorn server:app -b :5051 --workers 1 --threads 4 --timeout 300
```

For long audio or slow CPU inference, increase the timeout:

```bash
gunicorn server:app -b :5051 --workers 1 --threads 1 --timeout 3600
```

On native Windows, use `python run.py` or Waitress instead of Gunicorn.

## GPU inference

GPU inference requires an NVIDIA GPU and CUDA-compatible PyTorch. Docker GPU access also requires the NVIDIA Container Toolkit.

Set:

```bash
USE_CUDA=1
```

For Docker Compose, expose a GPU to the container:

```yaml
environment:
  USE_CUDA: "1"

deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Keep `WEB_WORKERS=1`: each worker loads a full copy of each active model. Scale with multiple containers or replicas instead of multiple workers per container.

## API usage

The primary endpoint is:

```text
POST /api/asr
```

It accepts `multipart/form-data` with:

- `audio`: Audio file to transcribe. Supported formats depend on `ffmpeg`; common formats include `wav`, `flac`, `ogg`, `mp3`, `m4a` and `webm`.
- `model_id`: Optional query parameter selecting a loaded model.
- `lang`: Optional query parameter selecting the default loaded model for a language.

Example:

```bash
curl -X POST \
  -F "audio=@sample.wav" \
  "http://localhost:5051/api/asr?model_id=nos_asr_gl_300m"
```

Typical response:

```json
{
  "model_id": "nos_asr_gl_300m",
  "model_type": "wav2vec2",
  "lang": "gl",
  "text": "ola mundo",
  "duration_s": 1.42
}
```

The `model_type` field is included when available in the loaded model record.

### Auxiliary endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Minimal web UI for uploading audio and viewing the transcription |
| `GET` | `/healthz` | Basic health check |
| `GET` | `/api/asr/models` | List loaded models grouped by language |
| `GET` | `/api/asr/check` | Validate a `model_id` or `lang` selection |

Examples:

```bash
curl http://localhost:5051/healthz
curl http://localhost:5051/api/asr/models
curl "http://localhost:5051/api/asr/check?model_id=nos_asr_gl_300m"
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ASR_API_CONFIG` | `config.json` | Path to the model registry |
| `MODELS_ROOT` | `models` | Root directory for model caches |
| `HF_HOME` | `models/.hf_cache` | Hugging Face cache directory |
| `HF_TOKEN` | unset | Hugging Face token for private or gated models |
| `USE_CUDA` | `0` | Set to `1` to enable CUDA inference |
| `MAX_AUDIO_MB` | `50` | Upload size limit in megabytes |
| `PORT` | `5052` for local launcher | Port used by `run.py` |
| `WEB_WORKERS` | `1` | Gunicorn workers |
| `WEB_THREADS` | `4` | Gunicorn threads |
| `WEB_TIMEOUT` | `300` | Request timeout in seconds |

## Long audio

The API accepts one audio file per request and returns the transcription only when the request finishes. It does not stream partial results through the HTTP API or the web UI.

For short and medium-length files, the audio can be submitted directly:

```bash
curl -X POST \
  -F "audio=@sample.wav" \
  "http://localhost:5051/api/asr?model_id=whisper_large_v3_turbo_gl_v1"

For long recordings, especially when using large models or CPU inference, it is often safer to split the audio before submitting it to the API. For example, to create 60-second chunks:

mkdir -p chunks
ffmpeg -i long_audio.wav -f segment -segment_time 60 -ac 1 -ar 16000 chunks/chunk_%03d.wav

Then process each chunk separately:

curl -X POST \
  -F "audio=@chunks/chunk_000.wav" \
  "http://localhost:5051/api/asr?model_id=whisper_large_v3_turbo_gl_v1"

For deployments that need to accept large files, increase both the upload limit and the request timeout:

MAX_AUDIO_MB=500
WEB_TIMEOUT=3600

Whisper can process audio internally in chunks, but a long recording submitted as a single HTTP request still produces a single response at the end. For production workflows with very long recordings, use external segmentation or implement a dedicated batch-processing endpoint.

## Demo page

Once the server is running, a simple web interface is available at:

- Docker Compose: <http://localhost:5051>
- Local launcher: <http://localhost:5052>

The page allows uploading an audio file and viewing the transcription returned by the API.

To customize the page:

- Edit `static/css/style.css` for styling.
- Edit `templates/index.html` for layout.

## Troubleshooting

### The API starts but a model is not available

Only models with `"load": true` are loaded at startup. Check:

```bash
curl http://localhost:5051/api/asr/models
```

Activate the desired model in `config.json` and restart the service.

### Whisper is slow or fails on long files

On CPU, Whisper is slow and memory-intensive. Use short files for testing and GPU for practical long-form inference. If the worker is killed with a message such as `SIGKILL` or `Perhaps out of memory?`, reduce the file length or move the workload to a GPU server.

### Docker build sends several gigabytes of context

Add or update `.dockerignore` so `models/`, audio files, chunks and archive files are not included in the build context.

### Docker cannot connect to the daemon

Start Docker Desktop or the Docker service, then verify:

```bash
docker info
```

On Windows with WSL2, `wsl --shutdown` followed by reopening Docker Desktop often resolves a stale engine state.

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Acknowledgements

This work is funded by the Ministerio para la Transformación Digital y de la Función Pública - Funded by EU – NextGenerationEU within the framework of the project Desarrollo de Modelos ALIA. Esta publicación del proyecto Desarrollo de Modelos ALIA está financiada por el Ministerio para la Transformación Digital y de la Función Pública y por el Plan de Recuperación, Transformación y Resiliencia – Financiado por la Unión Europea – NextGenerationEU.

Thanks also to [Dimensiona](https://www.dimensiona.com/gl/sobre-nos/) for the technical development of this API.
