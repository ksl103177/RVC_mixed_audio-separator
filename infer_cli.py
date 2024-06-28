import sys
import os
import logging
import soundfile as sf
from infer.modules.vc.modules import VC
from infer.modules.uvr5.modules import uvr
from configs.config import Config
from dotenv import load_dotenv
import traceback

# .env 파일 로드
load_dotenv('C:/RVC/RVC_test/.env')

# 설정 로드 및 VC 인스턴스 생성
config = Config()

# 모델 경로 설정
trained_model_name = "h100_isa_05.pth"  # 훈련된 모델 파일명
trained_model_path = os.path.join(trained_model_name)  # 훈련된 모델의 경로

# VC 인스턴스 생성 및 모델 로드
vc = VC(config)
vc.get_vc(trained_model_path)

# 로그 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def single_inference():
    spk_id = 0  # Speaker ID
    input_audio = "C:/RVC/RVC_test/test/test_02/output.wav"  # Path to input audio file
    transform = 0  # Pitch transform (semitones)
    f0_file = None  # Path to F0 file
    f0_method = "rmvpe"  # F0 extraction method
    file_index = "C:/RVC/RVC_test/logs/h100_isa_05/added_IVF1060_Flat_nprobe_1_h100_isa_05_v2.index"  # Path to feature index file
    index_rate = 0.75  # Index feature ratio
    filter_radius = 3  # Filter radius for median filtering
    resample_sr = 0  # Resample sample rate
    rms_mix_rate = 0.25  # RMS mix rate
    protect = 0.33  # Protection for consonants and breaths
    output_info = "C:/RVC/RVC_test/infer_output_log.txt"  # Path to output info file
    output_audio = "C:/RVC/RVC_test/infer_output/output_infer.wav"  # Path to output audio file

    logger.info("Starting single inference...")
    try:
        result, audio_output = vc.vc_single(
            spk_id,
            input_audio,
            transform,
            f0_file,
            f0_method,
            file_index,
            file_index,
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect
        )
        with open(output_info, "w") as f:
            f.write(result)
        # audio_output는 (샘플링 레이트, 오디오 데이터) 형태의 튜플
        sr, audio_data = audio_output
        sf.write(output_audio, audio_data, sr)
        logger.info("Single inference completed.")
    except Exception as e:
        logger.error(f"An error occurred during single inference: {e}")
        logger.error(traceback.format_exc())

def batch_inference():
    spk_id = 0  # Speaker ID
    input_dir = "path/to/input_dir"  # Path to input directory containing audio files
    output_dir = "path/to/output_dir"  # Path to output directory
    input_files = ["file1.wav", "file2.wav"]  # List of input files
    transform = 0  # Pitch transform (semitones)
    f0_method = "rmvpe"  # F0 extraction method
    file_index = "path/to/file_index.index"  # Path to feature index file
    index_rate = 0.75  # Index feature ratio
    filter_radius = 3  # Filter radius for median filtering
    resample_sr = 0  # Resample sample rate
    rms_mix_rate = 0.25  # RMS mix rate
    protect = 0.33  # Protection for consonants and breaths
    format = "wav"  # Output format
    output_info = "path/to/output_info.txt"  # Path to output info file

    logger.info("Starting batch inference...")
    try:
        results = vc.vc_multi(
            spk_id,
            input_dir,
            output_dir,
            input_files,
            transform,
            f0_method,
            file_index,
            file_index,
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect,
            format
        )
        with open(output_info, "w") as f:
            f.write("\n".join(results))
        logger.info("Batch inference completed.")
    except Exception as e:
        logger.error(f"An error occurred during batch inference: {e}")
        logger.error(traceback.format_exc())

def vocal_separation():
    model = "model_name"  # Model name
    input_dir = "path/to/input_dir"  # Path to input directory containing audio files
    output_vocal_dir = "path/to/output_vocal_dir"  # Path to output vocal directory
    input_files = ["file1.wav", "file2.wav"]  # List of input files
    output_instrumental_dir = "path/to/output_instrumental_dir"  # Path to output instrumental directory
    aggressiveness = 10  # Aggressiveness level
    format = "flac"  # Output format
    output_info = "path/to/output_info.txt"  # Path to output info file

    logger.info("Starting vocal separation...")
    try:
        results = uvr(
            model,
            input_dir,
            output_vocal_dir,
            input_files,
            output_instrumental_dir,
            aggressiveness,
            format
        )
        with open(output_info, "w") as f:
            f.write("\n".join(results))
        logger.info("Vocal separation completed.")
    except Exception as e:
        logger.error(f"An error occurred during vocal separation: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # 사용할 함수 호출
    # 예시: single_inference()
    single_inference()
    # 예시: batch_inference()
    # batch_inference()
    # 예시: vocal_separation()
    # vocal_separation()
