import moviepy.editor as mp
import json
from vosk import Model, KaldiRecognizer
import wave

def extract_audio_from_video(video_file, audio_file):
    video = mp.VideoFileClip(video_file)
    video.audio.write_audiofile(audio_file, codec='pcm_s16le')

def transcribe_audio_with_timestamps(audio_file, model_path):
    wf = wave.open(audio_file, "rb")
    recognizer = KaldiRecognizer(Model(model_path), wf.getframerate())
    recognizer.SetWords(True)
    transcription = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            if 'result' in result:
                transcription.extend(result['result'])

    # Обработка оставшихся данных
    result = json.loads(recognizer.FinalResult())
    if 'result' in result:
        transcription.extend(result['result'])

    return transcription

def main():
    video_file = r"C:\Users\admin\Desktop\hackaton 2024\test_recognize_people\speech\video.mkv"
    audio_file = "extracted_audio.wav"
    model_path = r"C:\Users\admin\Desktop\hackaton 2024\test_recognize_people\speech\vosk-model-small-ru-0.22"

    # Извлекаем аудио из видео
    extract_audio_from_video(video_file, audio_file)

    # Распознаем речь и получаем текст с таймингами
    transcription = transcribe_audio_with_timestamps(audio_file, model_path)

    for entry in transcription:
        print(f"Start: {entry['start']} End: {entry['end']} Text: {entry['word']}")

if __name__ == "__main__":
    main()
