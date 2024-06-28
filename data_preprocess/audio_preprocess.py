import glob
import librosa
from tqdm import tqdm
from scipy.io.wavfile import write
import numpy as np

def normalize_audio(file_path, max_wav_value=32768.0):
    data, sampling_rate = librosa.core.load(file_path, sr=None)
    data = data / np.abs(data).max() * 0.999  # 소리 크기 맞추기
    data = data * max_wav_value  # 데이터 범위를 max_wav_value에 맞추기
    data = data.astype(np.int16)  # 정수형으로 변환
    return data, sampling_rate

def process_files(base_path):
    for wav_file in tqdm(glob.glob(base_path + '/**/*.wav', recursive=True)):
        data, sampling_rate = normalize_audio(wav_file)
        write(wav_file, sampling_rate, data)

# 현재 디렉토리에서 시작
base_path = '/app/RVC/datasets/isa_speech copy'
process_files(base_path)
