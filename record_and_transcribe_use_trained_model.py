import os
import json
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pyaudio
import wave
from pynput import keyboard

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Sampling rate suitable for Whisper
CHUNK = 1024
BASE_OUTPUT_DIR = "dataset_sage"
WAVE_OUTPUT_TRAIN_DIR = os.path.join(BASE_OUTPUT_DIR, "train")
WAVE_OUTPUT_TEST_DIR = os.path.join(BASE_OUTPUT_DIR, "test")
WAVE_OUTPUT_VAL_DIR = os.path.join(BASE_OUTPUT_DIR, "val")

# Load model and setup pipeline for inference
model_path = "models_sage/whisper-large-finetune/"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path).to(device)
infer_pipe = pipeline("automatic-speech-recognition",
                      model=model,
                      tokenizer=processor.tokenizer,
                      feature_extractor=processor.feature_extractor,
                      device=device)

# Prompt the user to select the mode (train/test/val)
mode = input("Enter mode (train/test/val): ").strip().lower()
if mode == "train":
    WAVE_OUTPUT_DIR = WAVE_OUTPUT_TRAIN_DIR
    JSON_OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, "train.json")
elif mode == "test":
    WAVE_OUTPUT_DIR = WAVE_OUTPUT_TEST_DIR
    JSON_OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, "test.json")
elif mode == "val":
    WAVE_OUTPUT_DIR = WAVE_OUTPUT_VAL_DIR
    JSON_OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, "val.json")
else:
    print("Invalid mode. Defaulting to 'train'.")
    WAVE_OUTPUT_DIR = WAVE_OUTPUT_TRAIN_DIR
    JSON_OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, "train.json")

recording = False  # Flag to control recording
stop_program = False  # Flag to control when to stop the program
def on_press(key):
    global recording, stop_program
    try:
        if key.char == 'r':  # Press 'r' to start recording
            if not recording:
                print("Recording started. Press 's' to stop.")
                recording = True
        elif key.char == 's':  # Press 's' to stop recording
            if recording:
                print("Recording stopped.")
                recording = False
        elif key.char == 'q':  # Press 'q' to quit the program
            stop_program = True
            print("Quitting...")
            return False  # Stop listener
    except AttributeError:
        pass  # Handle special keys
def record_audio(filename):
    global recording, stop_program
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    while not stop_program:
        if recording:
            data = stream.read(CHUNK)
            frames.append(data)
        if not recording and frames:  # Stop recording if 's' was pressed
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save audio to .wav file only if recording was performed
    if frames:
        os.makedirs(WAVE_OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(WAVE_OUTPUT_DIR, filename)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        return filepath
    return None


# 在 transcribe_audio 函数中使用推理管道时，传入更多参数
def transcribe_audio(filepath):
    # 传入参数与 infer_test_all.py 保持一致
    generate_kwargs = {
        "task": "transcribe",
        "num_beams": 1,  # 可根据需要调整，通常 1-5 之间
        "language": "chinese",  # 或者根据需要设置成特定语言，如 "english" 等
    }

    # 使用推理管道进行转录并设置返回时间戳选项
    result = infer_pipe(filepath, return_timestamps=True, generate_kwargs=generate_kwargs)

    # 处理并组合转录的文本输出
    transcription = "".join([chunk["text"] for chunk in result["chunks"]])
    duration = result["chunks"][-1]["timestamp"][1] if "chunks" in result else 0
    return transcription, duration
def main():
    global stop_program
    with keyboard.Listener(on_press=on_press) as listener:
        while not stop_program:
            timestamp = int(time.time())
            audio_filename = f"record_{timestamp}.wav"
            audio_filepath = record_audio(audio_filename)

            # Only process if a recording was made
            if audio_filepath:
                transcription, duration = transcribe_audio(audio_filepath)

                # Print the transcription
                print(f"Transcription: {transcription}")

                # Prepare JSON data
                json_data = {
                    "audio": {"path": audio_filepath},
                    "sentence": transcription,
                    "duration": duration,
                    "sentences": [{"start": 0, "end": duration, "text": transcription}]
                }

                # Save JSON data to a file (write one line per JSON block)
                with open(JSON_OUTPUT_PATH, 'a') as json_file:
                    json.dump(json_data, json_file, ensure_ascii=False)
                    json_file.write('\n')

                print("Transcription saved. Press 'r' to record again or 'q' to quit.")
        listener.join()

if __name__ == "__main__":
    main()
