import tempfile

import whisper  # pyright: ignore
from moviepy import VideoFileClip  # type: ignore


def mp4_to_mp3(video_path: str, output_path: str):
    """
    Extracts audio from an MP4 directly into memory using PyAV,
    transcribes it using Whisper, and saves it to a Markdown file.
    """
    try:
        # Load the video clip
        video_clip = VideoFileClip(video_path)

        audio_clip = video_clip.audio

        assert audio_clip is not None, "No audio track found in the video."

        # Write the audio to a separate file
        audio_clip.write_audiofile(output_path)

        # Close the video and audio clips
        audio_clip.close()
        video_clip.close()
    except Exception as e:
        print(f"An error occurred: {e}")


def transcribe_audio(input_file: str):
    print(
        "Loading the large-v3 model... (This will take a moment, and will download ~2.9GB of weights on the first run)"
    )

    # Load the state-of-the-art large-v3 model.
    model = whisper.load_model("large-v3")

    # Transcribe the audio. Whisper automatically detects the language.
    result = model.transcribe(input_file)

    timestamps = "\n".join(
        f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}"  # type: ignore
        for segment in result["segments"]
    )
    return timestamps


def mp4_to_text(video_path: str, output_markdown_path: str):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        mp4_to_mp3(video_path, tmp.name)
        transcription = transcribe_audio(tmp.name)

    with open(output_markdown_path, "w") as f:
        f.write(transcription)


if __name__ == "__main__":
    INPUT_FILE = "test.mp4"
    TRANSCRIPTION_FILE = "test.txt"

    # mp4_to_mp3(INPUT_FILE, MP3_FILE)
    # transcribe_audio(MP3_FILE, TRANSCRIPTION_FILE)

    mp4_to_text(INPUT_FILE, TRANSCRIPTION_FILE)
