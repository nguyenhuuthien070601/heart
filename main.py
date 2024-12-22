from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
import librosa
import librosa.display
import cv2
from tensorflow.keras.models import load_model
import shutil
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Thêm middleware CORS vào ứng dụng FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép mọi domain truy cập
    allow_credentials=True,  # Cho phép gửi cookie nếu cần
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP
    allow_headers=["*"],  # Cho phép tất cả các header
)
app.mount("/", StaticFiles(directory="heartsound", html=True))
# Thư mục tạm thời để lưu trữ file
TEMP_DIRECTORY = "/tmp/uploads"

# Tạo thư mục tạm nếu chưa tồn tại
if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...), description: str = Form(...)):
    # Đường dẫn lưu file tạm thời
    file_location = f"{TEMP_DIRECTORY}/{file.filename}"

    # Lưu file vào thư mục tạm
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Trả về thông tin sau khi upload
    return {
        "info": f"File '{file.filename}' has been saved temporarily.",
        "description": description
    }

@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    # Đọc nội dung file
    content = await file.read()

    # Lưu tạm file vào thư mục /tmp
    file_path = os.path.join(TEMP_DIRECTORY, file.filename)
    with open(file_path, "wb") as f:
        f.write(content)

    # Xử lý file ngay
    results = detect(file_path)

    # Xóa file sau khi xử lý xong
    os.remove(file_path)

    # Phân tích kết quả
    result = "Unknown"
    count_dict = count_occurrences(results)
    total_count = len(results)
    percentage_dict = calculate_percentage(count_dict, total_count)
    most_common_value, percentage = most_frequent(percentage_dict)

    print(f"Giá trị xuất hiện nhiều nhất là {most_common_value} với {percentage:.2f}% xuất hiện trong mảng.")
    if most_common_value == 0:
        result = "Absent"
    elif most_common_value == 1:
        result = "Present"

    return {
        "results": result,
        "percentage": f"{percentage:.2f}%"
    }

# Hàm xử lý file và dự đoán (giữ nguyên như cũ)
def get_data(name_files, audio, time_limit, sr):
    n_fft = 512
    n_mels = 128
    hop_length = 256
    segment_length = time_limit * sr
    num_segments = (len(audio) // segment_length)
    x = []
    y = []
    for i in range(num_segments):
        start_sample = i * segment_length
        end_sample = (i + 1) * segment_length
        segment = audio[start_sample:end_sample]
        create_data(segment, n_fft, n_mels, hop_length)
        label = 1
        x.append(create_data(segment, n_fft, n_mels, hop_length))
        y.append(label)
    return x, y

def create_data(audio, n_fft, n_mels, hop_length):
    output_size = (75, 75, 1)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram_db = mel_spectrogram_db + np.abs(np.min(mel_spectrogram_db))
    mel_spectrogram_db = cv2.resize(mel_spectrogram_db, output_size[:2])
    mel_spectrogram_db = np.reshape(mel_spectrogram_db, output_size)
    return mel_spectrogram_db

def detect(name_files):
    duration = 2
    time_limit = duration
    sr = 4000
    x_test = []

    # Tải và tiền xử lý dữ liệu âm thanh
    audio, sr = librosa.load(name_files, sr=sr, offset=0)
    x, y = get_data(name_files, audio, time_limit=time_limit, sr=sr)

    # Thêm dữ liệu vào x_test và nhân bản
    x_test.extend(x)
    x_test = np.concatenate([x_test, x_test, x_test], axis=-1)

    # Tải mô hình và thực hiện dự đoán
    model = load_model('model/final_model_DenseNet1691.h5')
    predictions = model.predict(x_test)

    # Lưu kết quả vào mảng results
    results = []
    for result in predictions:
        print(np.argmax(result))
        results.append(np.argmax(result))
    print(results)

    # Trả về kết quả
    return results

def count_occurrences(results):
    # Tạo dictionary để đếm số lần xuất hiện của mỗi phần tử
    count_dict = {}

    # Đếm số lần xuất hiện của mỗi phần tử trong mảng
    for number in results:
        if number in count_dict:
            count_dict[number] += 1
        else:
            count_dict[number] = 1

    return count_dict

def calculate_percentage(count_dict, total_count):
    # Tính phần trăm số lần xuất hiện của mỗi giá trị
    percentage_dict = {key: (value / total_count) * 100 for key, value in count_dict.items()}
    return percentage_dict

def most_frequent(percentage_dict):
    # Tìm phần tử xuất hiện nhiều nhất
    most_common_value = max(percentage_dict, key=percentage_dict.get)
    return most_common_value, percentage_dict[most_common_value]
