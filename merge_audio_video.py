from moviepy import AudioFileClip, VideoFileClip  # type: ignore

# 1. Load the video and the audio file
video = VideoFileClip("media/videos/main/480p15/StepByStepQR.mp4")
audio = AudioFileClip("media/merged_audio.wav")

# 2. Set the audio of the video clip
# This creates a new object; it doesn't modify the original file yet
final_video = video.with_audio(audio)

# 3. Export the result
final_video.write_videofile("merged_output.mp4", codec="libx264", audio_codec="aac")
