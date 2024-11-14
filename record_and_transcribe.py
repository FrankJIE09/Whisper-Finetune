import os
import json
import time
import whisper
from pynput import keyboard
import pyaudio
import wave

# Initialize the Whisper model as a global variable
model = whisper.load_model("large")

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Sampling rate suitable for Whisper
CHUNK = 1024
BASE_OUTPUT_DIR = "dataset_sage"
WAVE_OUTPUT_TRAIN_DIR = os.path.join(BASE_OUTPUT_DIR, "train")
WAVE_OUTPUT_TEST_DIR = os.path.join(BASE_OUTPUT_DIR, "test")
WAVE_OUTPUT_VAL_DIR = os.path.join(BASE_OUTPUT_DIR, "val")

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

def transcribe_audio(filepath):
    global model  # Ensure we use the global model instance
    result = model.transcribe(filepath)
    transcription = result["text"]
    duration = result["segments"][-1]["end"] if "segments" in result and result["segments"] else 0
    return transcription, duration

def main():
    global stop_program
    # Start the keyboard listener for the entire program
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
                with open(JSON_OUTPUT_PATH, 'a') as json_file:  # Append to file
                    json.dump(json_data, json_file, ensure_ascii=False)
                    json_file.write('\n')  # Add a newline for readability

                print("Transcription saved. Press 'r' to record again or 'q' to quit.")
        listener.join()

if __name__ == "__main__":
    main()
