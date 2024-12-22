from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import cv2
from tensorflow.keras.models import load_model
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", StaticFiles(directory="heartsound", html=True))

@app.post("/upload_file/" )
async def upload_file(file: UploadFile = File(...)):
    try:
        # Đọc nội dung file trực tiếp từ UploadFile
        content = await file.read()

        # Chuyển nội dung file byte sang dữ liệu âm thanh
        audio, sr = librosa.load(librosa.util.buf_to_float(content), sr=4000, offset=0)

        # Gọi hàm detect để xử lý dữ liệu âm thanh
        results = detect(audio, sr)

        if not results:
            return {"results": "No results detected", "percentage": "N/A"}

        # Phân tích kết quả
        count_dict = count_occurrences(results)
        total_count = len(results)
        percentage_dict = calculate_percentage(count_dict, total_count)
        most_common_value, percentage = most_frequent(percentage_dict)

        # Xác định kết quả dự đoán
        result = "Unknown"
        if most_common_value == 0:
            result = "Absent"
        elif most_common_value == 1:
            result = "Present"

        return {
            "results": result,
            "percentage": f"{percentage:.2f}%"
        }

    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

def get_data(audio, time_limit, sr):
    n_fft = 512
    n_mels = 128
    hop_length = 256
    segment_length = time_limit * sr
    num_segments = len(audio) // segment_length
    x = []
    for i in range(num_segments):
        start_sample = i * segment_length
        end_sample = (i + 1) * segment_length
        segment = audio[start_sample:end_sample]
        x.append(create_data(segment, n_fft, n_mels, hop_length))
    return x

def create_data(audio, n_fft, n_mels, hop_length):
    output_size = (75, 75, 1)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram_db = mel_spectrogram_db + np.abs(np.min(mel_spectrogram_db))
    mel_spectrogram_db = cv2.resize(mel_spectrogram_db, output_size[:2])
    mel_spectrogram_db = np.reshape(mel_spectrogram_db, output_size)
    return mel_spectrogram_db

def detect(audio, sr):
    duration = 2
    x_test = []

    try:
        # Tiền xử lý dữ liệu âm thanh
        x = get_data(audio, time_limit=duration, sr=sr)
        x_test.extend(x)
        x_test = np.concatenate([x_test, x_test, x_test], axis=-1)

        # Tải mô hình và thực hiện dự đoán
        model = load_model('model/final_model_DenseNet1691.h5')
        predictions = model.predict(x_test)

        # Lưu kết quả vào mảng results
        results = [np.argmax(prediction) for prediction in predictions]
        return results
    except Exception as e:
        print(f"Error in detect: {e}")
        return []

def count_occurrences(results):
    count_dict = {}
    for number in results:
        count_dict[number] = count_dict.get(number, 0) + 1
    return count_dict

def calculate_percentage(count_dict, total_count):
    return {key: (value / total_count) * 100 for key, value in count_dict.items()}

def most_frequent(percentage_dict):
    most_common_value = max(percentage_dict, key=percentage_dict.get)
    return most_common_value, percentage_dict[most_common_value]
