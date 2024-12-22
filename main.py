from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
import librosa
import cv2
from tensorflow.keras.models import load_model
from fastapi.staticfiles import StaticFiles


app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/", StaticFiles(directory="heartsound", html=True))

# Sử dụng thư mục tạm thời `/tmp`
TMP_DIRECTORY = "/tmp"

# Tạo thư mục `/tmp` nếu chưa tồn tại (Railway thường đã có sẵn)
if not os.path.exists(TMP_DIRECTORY):
    os.makedirs(TMP_DIRECTORY)

@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    # Xác định đường dẫn file tạm
    file_path = os.path.join(TMP_DIRECTORY, file.filename)

    # Lưu file tạm vào thư mục `/tmp`
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        # Gọi hàm detect để phân tích file tạm
            results = detect(TMP_DIRECTORY + "/" + file.filename)
    
    finally:
        # Xóa file tạm sau khi xử lý
        if os.path.exists(file_path):
            os.remove(file_path)

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

def get_data(name_files, audio, time_limit, sr):
    n_fft = 512
    n_mels = 128
    hop_length = 256
    segment_length = time_limit * sr
    num_segments = (len(audio) // segment_length)
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

def detect(file_path):
    duration = 2
    sr = 4000
    x_test = []

    try:
        # Tải và tiền xử lý dữ liệu âm thanh
        audio, sr = librosa.load(file_path, sr=sr, offset=0)
        x = get_data(file_path, audio, time_limit=duration, sr=sr)
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
