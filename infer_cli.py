import sys
import os
import logging
import soundfile as sf
from infer.modules.vc.modules import VC
from infer.modules.uvr5.modules import uvr
from configs.config import Config
from dotenv import load_dotenv
import traceback

load_dotenv('your_.env_path')

config = Config()

trained_model_name = ""
trained_model_path = os.path.join(trained_model_name)

vc = VC(config)
vc.get_vc(trained_model_path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def single_inference():
    spk_id = 0
    input_audio = ""
    transform = 0
    f0_file = None
    f0_method = "rmvpe"
    file_index = ""
    index_rate = 0.75
    filter_radius = 3
    resample_sr = 0
    rms_mix_rate = 0.25
    protect = 0.33
    output_info = "infer_output_log.txt"
    output_audio = "infer_output/output_infer.wav"

    logger.info("단일 추론 시작...")
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
        sr, audio_data = audio_output
        sf.write(output_audio, audio_data, sr)
        logger.info("단일 추론 성공!")
    except Exception as e:
        logger.error(f"단일 추론 실패... : {e}")
        logger.error(traceback.format_exc())

def batch_inference():
    spk_id = 0
    input_dir = ""
    output_dir = ""
    input_files = ["file1.wav", "file2.wav"]
    transform = 0
    f0_method = "rmvpe"
    file_index = ""
    index_rate = 0.75
    filter_radius = 3
    resample_sr = 0
    rms_mix_rate = 0.25
    protect = 0.33
    format = "wav"
    output_info = "p"

    logger.info("다중 추론 시작...")
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
        logger.info("다중 추론 성공!")
    except Exception as e:
        logger.error(f"다중 추론 실패... : {e}")
        logger.error(traceback.format_exc())

def vocal_separation():
    model = ""
    input_dir = ""
    output_vocal_dir = ""
    input_files = ["file1.wav", "file2.wav"]
    output_instrumental_dir = ""
    aggressiveness = 10
    format = "flac"
    output_info = ""

    logger.info("보컬 분리 시작...")
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
        logger.info("보컬 분리 성공!")
    except Exception as e:
        logger.error(f"보컬 분리 실패... : {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    single_inference()
    # batch_inference()
    # vocal_separation()
