import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


class CaptionerError(Exception):
    """Raised when the captioning workflow cannot be completed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add burned-in captions to a video using OpenAI Whisper."
    )
    parser.add_argument(
        "video",
        type=str,
        nargs="?",
        help="Path to the input video file",
    )
    parser.add_argument(
        "--srt",
        type=str,
        default=None,
        help="Optional path for the generated SRT file (defaults to <video_stem>.srt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path for the captioned video (defaults to <video_stem>_captioned.ext)",
    )
    parser.add_argument(
        "--edit",
        action="store_true",
        help="Open interactive editor to edit subtitles after generation",
    )
    parser.add_argument(
        "--srt-only",
        action="store_true",
        help="Only generate SRT file, don't burn into video",
    )
    parser.add_argument(
        "--use-srt",
        type=str,
        default=None,
        help="Use existing SRT file instead of transcribing (skip transcription step)",
    )
    return parser.parse_args()


def ensure_prerequisites(api_key: str) -> None:
    if not api_key:
        raise CaptionerError(
            "OPENAI_API_KEY is not set. Export it before running this tool."
        )
    if shutil.which("ffmpeg") is None:
        raise CaptionerError(
            "ffmpeg is not available on PATH. Install ffmpeg and try again."
        )


def seconds_to_timestamp(value: float) -> str:
    total_ms = int(round(value * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert SRT timestamp format (HH:MM:SS,mmm) to seconds."""
    time_part, ms_part = timestamp.split(",")
    hours, minutes, seconds = map(int, time_part.split(":"))
    milliseconds = int(ms_part)
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0


def normalize_segments(raw_segments: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for segment in raw_segments:
        if segment is None:
            continue
        if hasattr(segment, "model_dump"):
            data = segment.model_dump()
        elif isinstance(segment, dict):
            data = segment
        else:
            data = segment.__dict__
        normalized.append(data)
    normalized.sort(key=lambda item: item.get("start", 0.0))
    return normalized


def chunk_segments(
    segments: Iterable[Dict[str, Any]], max_words: int = 3
) -> List[Dict[str, Any]]:
    chunked: List[Dict[str, Any]] = []
    for segment in segments:
        text = (segment.get("text") or "").strip()
        start = segment.get("start")
        end = segment.get("end")
        if not text or start is None or end is None:
            continue

        words = text.split()
        if not words:
            continue

        start_ts = float(start)
        end_ts = float(end)
        if end_ts <= start_ts:
            end_ts = start_ts + 0.6

        chunk_count = max(1, math.ceil(len(words) / max_words))
        chunk_duration = (end_ts - start_ts) / chunk_count

        for index in range(chunk_count):
            chunk_words = words[index * max_words : (index + 1) * max_words]
            if not chunk_words:
                continue
            chunk_start = start_ts + (index * chunk_duration)
            chunk_end = start_ts + ((index + 1) * chunk_duration)
            chunked.append(
                {
                    "start": chunk_start,
                    "end": chunk_end,
                    "text": " ".join(chunk_words),
                }
            )
    return chunked


def write_srt(segments: Iterable[Dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as srt_file:
        for index, segment in enumerate(segments, start=1):
            start = segment.get("start")
            end = segment.get("end")
            text = (segment.get("text") or "").strip()
            if start is None or end is None or not text:
                continue
            srt_file.write(f"{index}\n")
            srt_file.write(
                f"{seconds_to_timestamp(float(start))} --> "
                f"{seconds_to_timestamp(float(end))}\n"
            )
            srt_file.write(f"{text}\n\n")


def parse_srt(srt_path: Path) -> List[Dict[str, Any]]:
    """Parse an SRT file and return a list of segment dictionaries."""
    segments: List[Dict[str, Any]] = []
    if not srt_path.exists():
        raise CaptionerError(f"SRT file not found: {srt_path}")
    
    content = srt_path.read_text(encoding="utf-8")
    # Split by double newlines to get individual subtitle blocks
    blocks = re.split(r"\n\s*\n", content.strip())
    
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        
        try:
            # Skip the index line (first line)
            timestamp_line = lines[1]
            text_lines = lines[2:]
            
            # Parse timestamp: "HH:MM:SS,mmm --> HH:MM:SS,mmm"
            match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", timestamp_line)
            if not match:
                continue
            
            start_str, end_str = match.groups()
            start = timestamp_to_seconds(start_str)
            end = timestamp_to_seconds(end_str)
            text = "\n".join(text_lines).strip()
            
            if text:
                segments.append({"start": start, "end": end, "text": text})
        except (ValueError, IndexError):
            continue
    
    return segments


def interactive_edit_srt(srt_path: Path) -> None:
    """Interactive editor for editing SRT subtitle segments."""
    segments = parse_srt(srt_path)
    
    if not segments:
        print("No segments found in SRT file.")
        return
    
    print(f"\n{'='*70}")
    print(f"Interactive Subtitle Editor")
    print(f"{'='*70}")
    print(f"Editing: {srt_path}")
    print(f"Total segments: {len(segments)}")
    print(f"\nCommands:")
    print(f"  [Enter] - Keep current text and move to next")
    print(f"  [text]  - Edit the subtitle text")
    print(f"  'q'     - Quit and save changes")
    print(f"  's'     - Skip this segment (remove it)")
    print(f"  'b'     - Go back to previous segment")
    print(f"{'='*70}\n")
    
    edited_segments: List[Dict[str, Any]] = []
    index = 0
    
    while index < len(segments):
        segment = segments[index]
        start_ts = seconds_to_timestamp(segment["start"])
        end_ts = seconds_to_timestamp(segment["end"])
        
        print(f"\n[{index + 1}/{len(segments)}] {start_ts} --> {end_ts}")
        print(f"Current text: {segment['text']}")
        
        try:
            user_input = input("Edit (or Enter to keep, 'q' to quit, 's' to skip, 'b' to go back): ").strip()
            
            if user_input.lower() == "q":
                print("\nSaving changes and exiting editor...")
                break
            elif user_input.lower() == "s":
                print("Skipping this segment.")
                index += 1
                continue
            elif user_input.lower() == "b":
                if index > 0:
                    index -= 1
                    if edited_segments:
                        edited_segments.pop()
                    print("Going back to previous segment.")
                else:
                    print("Already at the first segment.")
                continue
            elif user_input == "":
                # Keep original text
                edited_segments.append(segment)
                index += 1
            else:
                # Update text
                edited_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": user_input
                })
                index += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted. Saving progress...")
            break
    
    # Add any remaining segments
    edited_segments.extend(segments[index:])
    
    # Write back to file
    write_srt(edited_segments, srt_path)
    print(f"\nSaved {len(edited_segments)} segments to {srt_path}")


def run_ffmpeg(args: List[str], context: str) -> None:
    try:
        subprocess.run(args, check=True)
    except FileNotFoundError as exc:
        raise CaptionerError("ffmpeg executable was not found.") from exc
    except subprocess.CalledProcessError as exc:
        raise CaptionerError(f"ffmpeg failed while attempting to {context}.") from exc


def extract_audio(video_path: Path, audio_path: Path) -> None:
    print("[1/5] Extracting audio with ffmpeg...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    run_ffmpeg(cmd, "extract audio from the video")


def transcribe_audio(audio_path: Path, client: OpenAI) -> Dict[str, Any]:
    print("[2/5] Transcribing audio with Whisper (whisper-1)...")
    try:
        with audio_path.open("rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
            )
    except Exception as exc:  # noqa: BLE001
        raise CaptionerError("Transcription failed. Check your API key and network.") from exc

    result: Dict[str, Any]
    if hasattr(transcription, "model_dump"):
        result = transcription.model_dump()
    elif isinstance(transcription, dict):
        result = transcription
    else:
        result = transcription.__dict__

    if "segments" not in result or not result["segments"]:
        raise CaptionerError(
            "Transcription completed but returned no timestamped segments."
        )
    return result


def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path) -> None:
    print("[4/5] Burning subtitles into the video...")
    style = (
        "FontName=Helvetica,"
        "FontSize=16,"
        "PrimaryColour=&H00FFFFFF&,"
        "OutlineColour=&H20303030&,"
        "BorderStyle=1,"
        "Outline=0.5,"
        "Shadow=0,"
        "Alignment=2,"
        "MarginV=60"
    )
    subtitles_filter = (
        f"subtitles=filename='{srt_path.as_posix()}':force_style='{style}'"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        subtitles_filter,
        "-c:a",
        "copy",
        str(output_path),
    ]
    run_ffmpeg(cmd, "burn subtitles into the video")


def process_video(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    api_key: str,
    use_existing_srt: Optional[Path] = None,
    edit_mode: bool = False,
    srt_only: bool = False,
) -> None:
    # If using existing SRT, skip transcription
    if use_existing_srt:
        print(f"[1/3] Using existing SRT file: {use_existing_srt}")
        segments = parse_srt(use_existing_srt)
        print(f"[2/3] Writing subtitles to {srt_path}...")
        write_srt(segments, srt_path)
        if edit_mode:
            interactive_edit_srt(srt_path)
        if not srt_only:
            print(f"[3/3] Burning subtitles into the video...")
            burn_subtitles(video_path, srt_path, output_path)
            print(f"Done! Captioned video saved to: {output_path}")
        else:
            print(f"Done! SRT file saved to: {srt_path}")
        return
    
    # Normal transcription workflow
    ensure_prerequisites(api_key)
    client = OpenAI(api_key=api_key)
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / f"{video_path.stem}_audio.wav"
        extract_audio(video_path, audio_path)
        result = transcribe_audio(audio_path, client)
        segments = chunk_segments(normalize_segments(result["segments"]), max_words=3)
        print(f"[3/5] Writing subtitles to {srt_path}...")
        write_srt(segments, srt_path)
        
        if edit_mode:
            interactive_edit_srt(srt_path)
        
        if not srt_only:
            burn_subtitles(video_path, srt_path, output_path)
            print(f"[5/5] Done! Captioned video saved to: {output_path}")
        else:
            print(f"[4/4] Done! SRT file saved to: {srt_path}")
    
    print(f"SRT file saved to: {srt_path}")


def main() -> None:
    # Load environment variables from .env file
    load_dotenv()
    
    args = parse_args()
    
    # If using existing SRT, video is optional if srt_only is True
    if args.use_srt:
        use_srt_path = Path(args.use_srt).expanduser().resolve()
        if not use_srt_path.exists():
            raise CaptionerError(f"SRT file not found: {use_srt_path}")
        
        if args.srt_only:
            # When srt_only, video is not needed
            srt_path = (
                Path(args.srt).expanduser().resolve()
                if args.srt
                else use_srt_path  # Default to same location as input
            )
            # Dummy paths for process_video (won't be used)
            video_path = Path("/dev/null")
            output_path = Path("/dev/null")
        else:
            # Need video to burn subtitles
            if not args.video:
                raise CaptionerError("Video path is required when using --use-srt (use --srt-only to skip video)")
            
            video_path = Path(args.video).expanduser().resolve()
            if not video_path.exists():
                raise CaptionerError(f"Input video not found: {video_path}")
            
            srt_path = (
                Path(args.srt).expanduser().resolve()
                if args.srt
                else video_path.with_name(f"{video_path.stem}.srt")
            )
            output_path = (
                Path(args.output).expanduser().resolve()
                if args.output
                else video_path.with_name(f"{video_path.stem}_captioned{video_path.suffix}")
            )
        
        api_key = os.environ.get("OPENAI_API_KEY") or ""
        process_video(
            video_path,
            srt_path,
            output_path,
            api_key,
            use_existing_srt=use_srt_path,
            edit_mode=args.edit,
            srt_only=args.srt_only,
        )
        return
    
    # Normal workflow requires video
    if not args.video:
        raise CaptionerError("Video path is required")
    
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise CaptionerError(f"Input video not found: {video_path}")

    srt_path = (
        Path(args.srt).expanduser().resolve()
        if args.srt
        else video_path.with_name(f"{video_path.stem}.srt")
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else video_path.with_name(f"{video_path.stem}_captioned{video_path.suffix}")
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    process_video(
        video_path,
        srt_path,
        output_path,
        api_key or "",
        edit_mode=args.edit,
        srt_only=args.srt_only,
    )


if __name__ == "__main__":
    try:
        main()
    except CaptionerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        sys.exit(1)

