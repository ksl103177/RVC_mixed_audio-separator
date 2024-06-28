import os
import librosa
import soundfile as sf
from multiprocessing import Pool
import glob
from tqdm import tqdm

def resample_audio(file_path):
    try:
        # 파일 읽기
        audio, sr = librosa.load(file_path, sr=None)  # 원본 샘플 레이트로 읽기
        # 22.05kHz로 리샘플
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        # 같은 파일명으로 저장
        sf.write(file_path, audio_resampled, 22050)
        return f"Processed: {file_path}"
    except Exception as e:
        return f"Error processing {file_path}: {e}"

def main():
    train_fl = glob.glob('/app/RVC/datasets/isa_speech copy/**/*.wav', recursive=True)
    
    print('멀티 프로세싱 시작...')
    # 멀티 프로세싱
    with Pool(processes=os.cpu_count()) as pool:
        # 진행 상태 표시를 위해 tqdm를 사용
        results = list(tqdm(pool.imap(resample_audio, train_fl), total=len(train_fl)))

        for result in results:
            print(result)

if __name__ == "__main__":
    main()
