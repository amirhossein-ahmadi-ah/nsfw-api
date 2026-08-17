# NSFW / NSFL Media Classification API

A small, production-oriented FastAPI service for classifying local images,
GIFs and videos with an ONNX model.

The project is designed around three priorities:

1. **High accuracy** — the model's original raw `[0, 255]` preprocessing is
   preserved; no accidental ImageNet normalization is applied.
2. **High speed** — long videos are **not decoded frame-by-frame**. The API
   probes duration with `ffprobe`, selects a small set of timestamps, seeks
   directly to those positions with FFmpeg and runs inference in small batches.
3. **Low resource usage** — no full-video frame list is stored in RAM, the
   model is loaded once per process, inference concurrency is bounded, and
   the number of analyzed frames has a hard upper bound.

> **Important:** this API accepts a local filesystem path, not an uploaded
> file body. That is intentional for low-copy local integrations such as a
> Telegram bot. Do not expose this endpoint directly to the public Internet
> unless you add an authenticated upload/API layer around it.

## Highlights

- FastAPI + Pydantic 2
- ONNX Runtime CPU inference
- FFmpeg + `ffmpeg-python` for video/GIF decoding
- `ffprobe` duration probing without decoding the media
- Bounded temporal sampling across the **whole** video
- No OpenCV dependency
- Lazy frame generation; no full video in memory
- Small ONNX inference batches for better CPU throughput
- Path allow-listing and symlink/traversal protection
- File-size and timeout limits
- Health, info and runtime statistics endpoints
- Cross-platform configuration (Windows / Linux / macOS)
- Optional systemd example for Linux
- GitHub-friendly repository layout with secrets and model binaries ignored

---

## How long-video processing works

This is the most important performance decision in the project.

A naive implementation might do:

```text
video -> decode every frame -> inspect every frame -> model
```

That is expensive for a 30-minute video.

This project does:

```text
video
  |
  +--> ffprobe -> duration only
  |
  +--> choose N timestamps
  |
  +--> FFmpeg input-seek -> decode one frame at each timestamp
  |
  +--> small inference batches
  |
  +--> aggregate scores
```

By default:

- `SAMPLE_INTERVAL_SECONDS=10`
- `MIN_MEDIA_FRAMES=8`
- `MAX_MEDIA_FRAMES=48`
- `INFERENCE_BATCH_SIZE=4`

So a 1-hour video is **not** sent through the model 216,000 times. It is
bounded to at most 48 sampled frames.

The exact number is configurable because there is an unavoidable trade-off:
more samples can improve temporal coverage, but increase CPU time.

### Why not use FFmpeg's `fps` filter?

A command such as `-vf fps=1/10` can reduce the number of frames sent to the
model, but FFmpeg may still need to decode the entire source stream to find
those frames. For long videos, that defeats the main performance goal.

Instead, this project performs **input-side seeking** for each selected
sample. FFmpeg jumps near the requested timestamp and decodes only the small
region needed for one output frame.

---

## Project structure

```text
nsfw-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application and endpoints
│   ├── config.py        # Environment-based configuration
│   ├── detector.py      # ONNX Runtime + batching + decision logic
│   ├── media.py         # FFmpeg/ffprobe media probing and sampling
│   ├── schemas.py       # API request/response models
│   └── security.py      # Path validation / allow-listing
├── files/               # Local files allowed to be classified
├── models/              # Model downloaded separately (ignored by Git)
├── logs/                # Runtime logs (ignored by Git)
├── systemd/             # Optional Linux service example
├── .env.example
├── .gitignore
├── benchmark.py
├── download_model.py
├── requirements.txt
├── telegram_client_example.py
├── test_api.py
└── README.md
```

---

# Installation

## 1. Requirements

- Python 3.11+ recommended
- FFmpeg with both `ffmpeg` and `ffprobe` available on `PATH`
- CPU with AVX-compatible Python/ONNX Runtime builds recommended
- Internet access during model download, unless the model is copied manually

GPU is not required. The default configuration uses ONNX Runtime CPU.

---

## 2. Install FFmpeg

### Windows

With `winget`:

```powershell
winget install Gyan.FFmpeg.Shared
```

Close and reopen PowerShell, then verify:

```powershell
ffmpeg -version
ffprobe -version
```

### macOS

With Homebrew:

```bash
brew install ffmpeg
```

Verify:

```bash
ffmpeg -version
ffprobe -version
```

### Debian / Ubuntu

```bash
sudo apt update
sudo apt install ffmpeg
```

Verify:

```bash
ffmpeg -version
ffprobe -version
```

### Other Linux distributions

Install the FFmpeg package from your distribution and make sure both
executables are available through `PATH`.

If FFmpeg is installed somewhere custom, set:

```dotenv
FFMPEG_BINARY=/path/to/ffmpeg
FFPROBE_BINARY=/path/to/ffprobe
```

On Windows, for example:

```dotenv
FFMPEG_BINARY=C:\\ffmpeg\\bin\\ffmpeg.exe
FFPROBE_BINARY=C:\\ffmpeg\\bin\\ffprobe.exe
```

---

## 3. Create a Python environment

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Configure the project

Copy `.env.example` to `.env`.

Windows:

```powershell
Copy-Item .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

The default configuration uses `./files` as the allowed directory and
`./models/image-safety-classifier-l.onnx` as the model path, so the repository
can be moved between machines without editing Linux-specific absolute paths.

For a local bot integration, you can add its download directory to
`ALLOWED_DIRS`, comma-separated:

```dotenv
ALLOWED_DIRS=./files,C:\\my-bot\\downloads
```

---

## 5. Download the model

```bash
python download_model.py
```

The model is downloaded from:

```text
OwenElliott/image-safety-classifier-l
```

and stored as:

```text
models/image-safety-classifier-l.onnx
```

The model file is intentionally excluded from Git because it is a large
binary artifact. Each deployment can download it separately.

---

# Run

From the project root:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1
```

The API is intentionally bound to `127.0.0.1` by default.

Do **not** use multiple workers on a small machine unless you deliberately
want multiple copies of the ONNX model in RAM.

On Windows, the same command works in PowerShell after activating `.venv`.

---

# API

## `GET /health`

Lightweight health check.

```bash
curl http://127.0.0.1:8001/health
```

Example:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

## `GET /info`

Returns model, thresholds, supported formats and sampling configuration.

## `GET /stats`

Returns process RSS, CPU count, inference activity and uptime.

## `POST /detect`

Request:

```json
{
  "path": "./files/example.mp4"
}
```

The file must be inside one of the configured `ALLOWED_DIRS` directories.

The response contains:

- `is_nsfw`
- `is_nsfl`
- `nsfw`
- `nsfl`
- `sfw`
- `checked_frames`
- `suspicious_frames`
- `suspicious_ratio`
- `max_nsfw`
- `average_nsfw`
- `max_consecutive_suspicious`
- `decision_reason`
- `processing_time_ms`

Interactive API documentation is available at:

```text
http://127.0.0.1:8001/docs
http://127.0.0.1:8001/redoc
```

---

# Supported media

### Images

- JPG / JPEG
- PNG
- WebP
- BMP

Images are classified as one frame.

### Animated media

- GIF

GIFs are sampled using the same bounded timestamp-seeking approach as video.

### Video

- MP4
- MOV
- MKV
- WebM
- AVI

The actual codec support depends on the installed FFmpeg build.

---

# Accuracy / preprocessing

The model expects RGB pixels resized to `224x224` and represented as float32
values in the original `[0, 255]` range.

The API intentionally does **not** do:

```python
image / 255.0
```

and does not apply ImageNet mean/std normalization.

Changing this preprocessing can materially change the model's output and can
produce false negatives. The current implementation keeps the original
expected preprocessing.

For video, FFmpeg performs the resize before the frame reaches Python. This
avoids an additional Pillow resize for every sampled frame.

---

# Performance tuning

The most important settings are:

| Setting | Default | Purpose |
|---|---:|---|
| `SAMPLE_INTERVAL_SECONDS` | `10` | Approximate temporal spacing |
| `MIN_MEDIA_FRAMES` | `8` | Minimum video/GIF samples |
| `MAX_MEDIA_FRAMES` | `48` | Hard maximum samples |
| `INFERENCE_BATCH_SIZE` | `4` | ONNX CPU batch size |
| `MAX_CONCURRENT_INFERENCE` | `1` | Concurrent inference slots |
| `MAX_FILE_SIZE_MB` | `100` | Input file limit |
| `DETECTION_TIMEOUT` | `60` | Overall request timeout |

### Fast / low CPU profile

```dotenv
SAMPLE_INTERVAL_SECONDS=15
MIN_MEDIA_FRAMES=6
MAX_MEDIA_FRAMES=32
INFERENCE_BATCH_SIZE=4
MAX_CONCURRENT_INFERENCE=1
```

### Balanced profile

```dotenv
SAMPLE_INTERVAL_SECONDS=10
MIN_MEDIA_FRAMES=8
MAX_MEDIA_FRAMES=48
INFERENCE_BATCH_SIZE=4
MAX_CONCURRENT_INFERENCE=1
```

### Higher coverage profile

```dotenv
SAMPLE_INTERVAL_SECONDS=5
MIN_MEDIA_FRAMES=12
MAX_MEDIA_FRAMES=96
INFERENCE_BATCH_SIZE=4
MAX_CONCURRENT_INFERENCE=1
```

There is no configuration that can simultaneously maximize accuracy, minimize
CPU, and inspect every moment of every video. The project therefore makes the
sampling trade-off explicit and bounded instead of silently decoding an
entire long video.

For a moderation bot, the **balanced profile** is a good starting point.
Benchmark on your actual media before changing it.

---

# Security

This service accepts filesystem paths, so path security is important.

The API:

- resolves paths before checking them
- rejects `..` traversal
- rejects symlink escapes outside the allow-list
- rejects directories
- rejects unsupported extensions
- enforces a maximum file size
- binds to localhost by default

If you expose this service over a network, add authentication and preferably
change the architecture to accept controlled uploads instead of arbitrary
filesystem paths.

Do **not** put `.env`, model binaries, logs, or user media into a public Git
repository.

---

# Testing

Start the API first, then:

```bash
python test_api.py ./files/test.jpg ./files/test.gif ./files/test.mp4
```

Basic checks:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/info
curl http://127.0.0.1:8001/stats
```

Benchmark:

```bash
python benchmark.py ./files/test.mp4 10
```

For a useful benchmark, test several short and long videos and compare:

- wall-clock latency
- `checked_frames`
- CPU usage
- RSS memory
- classification results

---

# Telegram / local Python integration

`telegram_client_example.py` contains a minimal async client using `httpx`.
The client sends only a local path, so the media does not get uploaded twice.

Install `httpx` in the bot's environment and adapt the example to your bot.

The recommended failure policy is to treat an unavailable classifier as
**unknown**, not automatically SFW.

---

# Linux systemd

The repository includes an optional example service under `systemd/`.
It is not required on Windows or macOS.

The service uses one worker because every additional worker loads another
copy of the ONNX model.

---

# GitHub publishing checklist

Before pushing:

```bash
# Make sure secrets/local config are not tracked
git status

# Verify Python files compile
python -m compileall app *.py

# Verify FFmpeg is available
ffmpeg -version
ffprobe -version
```

The repository `.gitignore` excludes:

- `.env`
- virtual environments
- logs
- local media
- ONNX model binaries
- Python caches
- IDE/build artifacts

Commit `.env.example`, not `.env`.

---

# License / model notice

The application code and the model are separate artifacts. Before publishing
or redistributing a deployment, review the license and terms of the ONNX model
and its upstream repository.
