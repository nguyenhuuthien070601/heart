from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import librosa.display
import cv2
from tensorflow.keras.models import load_model



model = load_model('model/final_model_DenseNet1691.h5')
sample_data = np.random.rand(1, 75, 75, 1)  # Thay đổi kích thước nếu cần
try:
    print("Sample prediction:", model.predict(sample_data))
except Exception as e:
    print("Error in model prediction:", str(e))
