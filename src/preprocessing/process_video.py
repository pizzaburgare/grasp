"""Video processing: transcription and frame description extraction."""

import base64
import io
import tempfile
from collections.abc import Callable

import numpy as np
import whisper  # pyright: ignore
from langchain_core.messages import HumanMessage, SystemMessage
from moviepy import VideoFileClip  # type: ignore
from PIL import Image

from src.core.llm_metrics import LLMUsage, combine_llm_usage, extract_llm_usage, make_openrouter_llm

type Frame = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
type FrameSelector = Callable[[VideoFileClip], list[tuple[Frame, float]]]


def _extract_audio(video_path: str, output_path: str) -> None:
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    assert audio_clip is not None, "No audio track found in the video."
    audio_clip.write_audiofile(output_path)
    audio_clip.close()
    video_clip.close()


type TranscriptSegment = tuple[float, str]  # (timestamp, sentence)


def _transcribe(audio_path: str) -> list[TranscriptSegment]:
    """Transcribe audio and return list of (timestamp, sentence) tuples."""
    print("Loading the large-v3 model... (downloads ~2.9GB on first run)")
    model = whisper.load_model("large-v3")
    result = model.transcribe(audio_path)
    return [
        (float(segment["start"]), segment["text"].strip())  # type: ignore
        for segment in result["segments"]
    ]


def uniform_frame_selector(interval_seconds: float = 5.0) -> FrameSelector:
    """Returns a frame selector that picks frames at uniform intervals."""

    def selector(video: VideoFileClip) -> list[tuple[Frame, float]]:
        duration = video.duration
        timestamps = np.arange(0, duration, interval_seconds).tolist()
        return [(video.get_frame(t), t) for t in timestamps]  # type: ignore[misc]

    return selector


def transcription_frame_selector(transcription: list[TranscriptSegment]) -> FrameSelector:
    """Returns a frame selector that picks frames at transcription segment starts."""
    timestamps = [ts for ts, _ in transcription]

    def selector(video: VideoFileClip) -> list[tuple[Frame, float]]:
        return [(video.get_frame(t), t) for t in timestamps]  # type: ignore[misc]

    return selector


def _describe_frame(
    frame: Frame, timestamp: float, model: str = "google/gemini-2.0-flash-001"
) -> tuple[LLMUsage | None, str]:
    """Returns an LLM description of a frame."""
    try:
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        data = base64.b64encode(buf.getvalue()).decode()

        llm = make_openrouter_llm(model=model, title="Video Frame Describer")

        messages = [
            SystemMessage(content="You are a helpful assistant that describes images."),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{data}"},
                    },
                    {"type": "text", "text": "Describe this image."},
                ]
            ),
        ]

        response = llm.invoke(messages)
        usage = extract_llm_usage(response)

    except Exception as e:  # noqa: BLE001
        print(f"An error occurred while processing the frame: {e}")
        return (None, f"Error processing image at {timestamp:.2f}s")
    return (usage, f"Description of image at {timestamp:.2f}s: {response.content}")


def _merge_transcription_and_frames(
    transcription: list[TranscriptSegment],
    frame_descriptions: list[tuple[float, str]],
) -> str:
    """Merge transcription segments and frame descriptions chronologically."""
    combined: list[tuple[float, str, str]] = []
    for ts, text in transcription:
        combined.append((ts, "transcript", text))
    for ts, desc in frame_descriptions:
        combined.append((ts, "frame", desc))

    combined.sort(key=lambda x: x[0])

    lines: list[str] = []
    for ts, kind, content in combined:
        if kind == "transcript":
            lines.append(f"[{ts:.2f}s] {content}")
        else:
            lines.append(f"[{ts:.2f}s] [VISUAL] {content}")
    return "\n".join(lines)


def mp4_to_text(
    video_path: str,
    output_path: str,
    frame_selector: FrameSelector | None = None,
) -> LLMUsage | None:
    """
    Converts an MP4 to a text file containing timestamped transcription and
    VLM frame descriptions. Uses a temp file for intermediate audio.

    Args:
        video_path: Path to input video file.
        output_path: Path for output text file.
        frame_selector: Algorithm to select frames. Takes a VideoFileClip and returns
            list of (frame, timestamp) tuples. Defaults to transcription-based selection.
    """
    # Transcribes audio
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        _extract_audio(video_path, tmp.name)
        transcription = _transcribe(tmp.name)

    # If none is given, select frames at transcription segment timestamps
    if frame_selector is None:
        frame_selector = uniform_frame_selector(interval_seconds=30.0)

    video = VideoFileClip(video_path)
    frames_with_timestamps = frame_selector(video)
    video.close()

    usages: list[LLMUsage | None] = []
    frame_descriptions: list[tuple[float, str]] = []
    for frame, timestamp in frames_with_timestamps:
        usage, description = _describe_frame(frame, timestamp)
        usages.append(usage)
        frame_descriptions.append((timestamp, description))

    output = _merge_transcription_and_frames(transcription, frame_descriptions)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    return combine_llm_usage(usages)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe an MP4 video to text with frame descriptions."
    )
    parser.add_argument("input", help="Path to the input MP4 file")
    parser.add_argument("output", help="Path to the output TXT file")
    args = parser.parse_args()

    mp4_to_text(args.input, args.output)
