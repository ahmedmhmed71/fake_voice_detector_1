import subprocess
import os
import tensorflow as tf
import pickle
import json
import numpy as np
import librosa
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#تحويل إلى wav
def convert_to_wav(input_path):
    output_path = input_path + ".wav"

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_path

# تحميل الملفات
model = tf.keras.models.load_model(os.path.join(BASE_DIR, "model.keras"))

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "config.json"), "r") as f:
    config = json.load(f)

SR = config["sr"]
N_MELS = config["n_mels"]
MAX_FRAMES = config["max_frames"]
N_FFT = config["n_fft"]
HOP_LENGTH = config["hop_length"]

# نفس دالة استخراج الميّل عندك
def extract_melspec(file_path):
    y, sr = librosa.load(file_path, sr=SR)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    # ضبط الطول
    if mel.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)))
    else:
        mel = mel[:, :MAX_FRAMES]

    return mel

# دالة التوقع
# دالة التوقع
def predict_audio(file_path):

    spec = extract_melspec(file_path)

    spec = spec.reshape(1, -1)
    spec = scaler.transform(spec)
    spec = spec.reshape(1, N_MELS, MAX_FRAMES, 1)

    prob = model.predict(spec)[0][0]

    # حذف الملف بعد الاستخدام (اختياري)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except:
        pass

    return {
        "label": "Fake" if prob >= 0.5 else "Real",
        "confidence": float(prob)
    }

# تجرب



