import os
import moviepy.editor as mp
import whisper
import wave
import json

# Установите путь к ffmpeg
ffmpeg_path = r"C:\Users\admin\Desktop\hackaton 2024\test_recognize_people\ffmpeg-master-latest-win64-gpl\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

def extract_audio_from_video(video_file, audio_file):
    video = mp.VideoFileClip(video_file)
    video.audio.write_audiofile(audio_file, codec='pcm_s16le')

def transcribe_audio_with_whisper(audio_file):
    model = whisper.load_model("tiny")  # Используйте модель, подходящую по размеру и качеству (small, medium, large) Сделать выборку
    result = model.transcribe(audio_file)
    transcription = []

    for segment in result['segments']:
        transcription.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text']
        })

    return transcription

def main():
    video_file = r"C:\Users\admin\Desktop\hackaton 2024\test_recognize_people\speech\video.mkv"
    audio_file = "extracted_audio1.wav"

    # Извлекаем аудио из видео
    extract_audio_from_video(video_file, audio_file)

    # Распознаем речь и получаем текст с таймингами по предложениям
    transcription = transcribe_audio_with_whisper(audio_file)

    for entry in transcription:
        print(f"Start: {entry['start']} End: {entry['end']} Text: {entry['text']}")


if __name__ == "__main__":
    main()