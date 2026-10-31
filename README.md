# ClipForge

ClipForge is a complete YouTube-to-short-form video studio. Paste a public YouTube URL, choose the number and duration of clips, and the worker downloads, transcribes, scores, reframes, captions, and renders ready-to-post 9:16 MP4s.

## What it does

- Uses `yt-dlp` to download a public YouTube source.
- Extracts mono 16 kHz audio with FFmpeg.
- Chunks long audio and transcribes with Groq `whisper-large-v3` word timestamps.
- Sends the timestamped transcript to Groq `llama-3.3-70b-versatile` and requests strict JSON moments.
- Ranks moments by virality, rejects overlaps, and honestly returns fewer clips when the transcript does not contain enough strong moments.
- Renders each clip with FFmpeg, a blurred background fit, a portrait foreground, and an animated word-by-word ASS caption track with the current word highlighted.
- Runs a lightweight OpenCV Haar face scan to identify a stable speaker center before rendering.
- Uses a persistent SQL job/clip table, a background thread pool, SSE progress events, history, retry, delete, per-clip downloads, ZIP downloads, and timing edits.
- Removes files older than 24 hours at startup.

## Requirements

- Python 3.11+
- Node.js 20+
- FFmpeg with libass support (`ffmpeg -version` should list `--enable-libass` for burned-in captions)
- A Groq API key
- MySQL 8+ for production. SQLite is supported automatically for local development.

## Environment

Copy `.env.example` to `.env` and set:

```bash
GROQ_API_KEY=gsk_...
# Optional production database. Leave unset to use ./clipforge.db.
DATABASE_URL=mysql+pymysql://clipforge:password@127.0.0.1:3306/clipforge
```

Never expose `GROQ_API_KEY` in the frontend.

## Local run

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
export GROQ_API_KEY='gsk_...'
uvicorn backend.main:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080`.

For development with hot reload:

```bash
cd frontend && npm run dev
# in another terminal:
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8080
```

## API surface

- `POST /api/jobs` — create a job
- `GET /api/jobs` — history
- `GET /api/jobs/:id` — job plus clips
- `GET /api/jobs/:id/events` — SSE progress stream
- `POST /api/jobs/:id/retry` — retry a failed job
- `DELETE /api/jobs/:id` — delete metadata and files
- `POST /api/clips/:id/rerender` — edit start/end and render again
- `GET /api/clips/:id/download` — individual MP4
- `GET /api/jobs/:id/zip` — ZIP of ready MP4s
- `GET /api/health` — environment readiness

## Important operational notes

YouTube downloads are subject to YouTube availability, copyright, age restrictions, sign-in requirements, and regional access. ClipForge surfaces download failures instead of manufacturing output. Groq rate limits are returned as job errors. Files are stored under `storage/` and cleaned after 24 hours.

For production, run behind TLS with a process manager and use MySQL. The included SQLite fallback is intended for a single-node deployment or local testing.
