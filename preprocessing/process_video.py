import base64
import io
import os
import tempfile

import whisper  # pyright: ignore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from moviepy import VideoFileClip  # type: ignore
from PIL import Image
from pydantic import SecretStr

from src.llm_metrics import extract_llm_usage


def _extract_audio(video_path: str, output_path: str):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    assert audio_clip is not None, "No audio track found in the video."
    audio_clip.write_audiofile(output_path)
    audio_clip.close()
    video_clip.close()


def _transcribe(audio_path: str) -> str:
    print("Loading the large-v3 model... (downloads ~2.9GB on first run)")
    model = whisper.load_model("large-v3")
    result = model.transcribe(audio_path)
    return "\n".join(
        f"{segment['start']:.2f}s -> {segment['end']:.2f}s --- {segment['text']}"  # type: ignore
        for segment in result["segments"]
    )


def _parse_start_times(transcription: str) -> list[float]:
    return [float(line.split("s ->")[0].strip()) for line in transcription.splitlines() if "s ->" in line]


def _describe_frame(video_path: str, timestamp: float, model: str = "google/gemini-2.0-flash-001") -> tuple[float, str]:
    """Extracts a frame at the given timestamp and returns an LLM description."""
    try:
        video_clip = VideoFileClip(video_path)
        frame = video_clip.get_frame(timestamp)
        video_clip.close()

        img = Image.fromarray(frame)  # type: ignore[arg-type]
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        data = base64.b64encode(buf.getvalue()).decode()

        llm = ChatOpenAI(
            model=model,
            api_key=SecretStr(os.getenv("OPENROUTER_API_KEY") or ""),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Video Frame Describer",
            },
        )

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
        cost = extract_llm_usage(response).cost_usd

        if cost is None:
            cost = 0.0

        return (cost, f"Description of image at {timestamp:.2f}s: {response.content}")
    except Exception as e:
        print(f"An error occurred while processing the frame: {e}")
        return (0.0, f"Error processing image at {timestamp:.2f}s")


def mp4_to_text(video_path: str, output_path: str) -> float:
    """
    Converts an MP4 to a text file containing timestamped transcription and
    VLM frame descriptions for each segment. Uses a temp file for intermediate audio.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        _extract_audio(video_path, tmp.name)
        transcription = _transcribe(tmp.name)

    lines = transcription.splitlines()
    times = _parse_start_times(transcription)

    total_cost = 0.0
    output = ""
    for time, line in zip(times, lines, strict=False):
        output += f"{line.strip()}\n"
        cost, desctiption = _describe_frame(video_path, time)
        total_cost += cost
        output += desctiption + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    return total_cost


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe an MP4 video to text with frame descriptions.")
    parser.add_argument("input", help="Path to the input MP4 file")
    parser.add_argument("output", help="Path to the output TXT file")
    args = parser.parse_args()

    mp4_to_text(args.input, args.output)
