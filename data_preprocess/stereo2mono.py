import os
import glob
import soundfile as sf
import librosa
from tqdm import tqdm

def is_stereo(file_path):
    with sf.SoundFile(file_path) as f:
        return f.channels == 2

def convert_to_mono(file_path, output_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    sf.write(output_path, y, sr)

def process_files(base_path):
    for wav_file in tqdm(glob.glob(base_path + '/**/*.wav', recursive=True)):
        if is_stereo(wav_file):
            print(f"Converting {wav_file} to mono")
            convert_to_mono(wav_file, wav_file)  # Overwrite the file

# 현재 디렉토리에서 시작
base_path = '/app/RVC/datasets/isa_speech copy'
process_files(base_path)