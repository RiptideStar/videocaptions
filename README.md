# Video Captioning CLI

Python command-line helper that extracts audio from a video, transcribes it
through OpenAI Whisper (`whisper-1`), produces an SRT subtitle file, and burns
styled captions back into the video using `ffmpeg`.

## Prerequisites

- Python 3.9+
- `ffmpeg` available on your `PATH`
- OpenAI API key with access to Whisper (`OPENAI_API_KEY`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your OpenAI API key:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

Or manually create `.env` and add:
```
OPENAI_API_KEY=sk-your-api-key-here
```

The script will automatically load the API key from the `.env` file when you run it.

## Usage

```bash
python captioner.py /path/to/video.mp4
```

Default behavior:

- Writes an SRT file next to the input video (e.g., `video.srt`).
- Creates a captioned copy named `video_captioned.mp4` with:
  - White text
  - Black, semi-opaque background box
  - Bottom-center positioning

### Optional flags

- `--srt /custom/output.srt` – override the location of the SRT file.
- `--output /custom/video_captioned.mp4` – override the captioned video path.

## Troubleshooting

- Ensure `ffmpeg` is installed (`brew install ffmpeg` on macOS).
- Confirm `OPENAI_API_KEY` is set in your `.env` file in the project root.
- The script prints progress for each major step; if it stalls, re-run with
  `FFMPEG`'s own verbose logs to inspect codec issues.


