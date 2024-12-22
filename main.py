from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import librosa.display
import cv2
from tensorflow.keras.models import load_model

app = FastAPI()

# Thêm middleware CORS vào ứng dụng FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép mọi domain truy cập
    allow_credentials=True,  # Cho phép gửi cookie nếu cần
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP
    allow_headers=["*"],  # Cho phép tất cả các header
)

@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    # Đọc nội dung file từ bộ nhớ
    content = await file.read()

    # Tiền xử lý và phân tích nội dung âm thanh trực tiếp
    try:
        # Chuyển đổi nội dung byte thành dữ liệu âm thanh
        audio, sr = librosa.load(librosa.util.buf_to_float(content), sr=4000, offset=0)
    except Exception as e:
        return {"error": "Error processing audio file.", "details": str(e)}

    # Tiền xử lý và dự đoán
    duration = 2
    time_limit = duration
    x, y = get_data(audio, time_limit=time_limit, sr=sr)

    # Chuẩn bị dữ liệu dự đoán
    x_test = np.concatenate([x, x, x], axis=-1)

    # Tải mô hình và thực hiện dự đoán
    model = load_model('model/final_model_DenseNet1691.h5')
    predictions = model.predict(x_test)

    # Xử lý kết quả dự đoán
    results = [np.argmax(prediction) for prediction in predictions]
    count_dict = count_occurrences(results)
    total_count = len(results)
    percentage_dict = calculate_percentage(count_dict, total_count)
    most_common_value, percentage = most_frequent(percentage_dict)

    # Đặt kết quả dựa trên giá trị phổ biến nhất
    result = "Unknown"
    if most_common_value == 0:
        result = "Absent"
    elif most_common_value == 1:
        result = "Present"

    return {
        "results": result,
        "percentage": f"{percentage:.2f}%"
    }

# Hàm xử lý file và dự đoán
def get_data(audio, time_limit, sr):
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
        x.append(create_data(segment, n_fft, n_mels, hop_length))
        y.append(1)  # Gán nhãn mặc định
    return x, y

def create_data(audio, n_fft, n_mels, hop_length):
    output_size = (75, 75, 1)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram_db = mel_spectrogram_db + np.abs(np.min(mel_spectrogram_db))
    mel_spectrogram_db = cv2.resize(mel_spectrogram_db, output_size[:2])
    mel_spectrogram_db = np.reshape(mel_spectrogram_db, output_size)
    return mel_spectrogram_db

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
