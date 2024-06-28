import sys
import os
import soundfile as sf
import time
import shutil
import traceback
import yaml

from audio_separator.separator import Separator
from pydub import AudioSegment
from infer.modules.vc.modules import VC
from infer.modules.uvr5.modules import uvr
from configs.config import Config
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv('C:/RVC/RVC_test/.env')

# YAML 설정 파일 로드
with open('C:/RVC/RVC_test/load_yaml/rvc_main_config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

# 설정 로드 및 VC 인스턴스 생성
config = Config()

# 모델 경로 설정
trained_model_name = config_data['model']['trained_model_name']
trained_model_path = os.path.join(trained_model_name)

# VC 인스턴스 생성 및 모델 로드
vc = VC(config)
vc.get_vc(trained_model_path)

def move_file(file_path, target_path):
    try:
        if os.path.isfile(file_path):
            shutil.move(file_path, target_path)
            print(f"File '{file_path}' has been moved to '{target_path}'.")
        else:
            print(f"No such file: '{file_path}'")
    except Exception as e:
        print(f"Error moving file '{file_path}' to '{target_path}': {e}")

def delete_file(file_path):
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"File '{file_path}' has been deleted.")
        else:
            print(f"No such file: '{file_path}'")
    except Exception as e:
        print(f"Error deleting file '{file_path}': {e}")

def audio_sep(input_audio_path, first_mr_output_dir):
    start_time = time.time()
    
    input_extension = os.path.splitext(input_audio_path)[1].lower()
    supported_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.mp4']
    
    if input_extension not in supported_extensions:
        print(f"지원하지 않는 파일 형식입니다: {input_extension}")
        return None, None
    
    if input_extension != '.wav':
        audio = AudioSegment.from_file(input_audio_path)
        input_audio_path = input_audio_path.replace(input_extension, '.wav')
        audio.export(input_audio_path, format='wav')
    
    separator = Separator()
    trash_file_list = []

    print("UVR_MDXNET_Main 모델 로드 및 분리 시작...")
    separator.load_model('UVR_MDXNET_Main.onnx')
    output_file_paths_1 = separator.separate(input_audio_path)
    print(f"UVR_MDXNET_Main 모델로 분리 완료! 다음과 같이 분리 되었습니다. : {output_file_paths_1}")
    
    mr_file = output_file_paths_1[0]
    vocal_file = output_file_paths_1[1]
    
    os.makedirs(first_mr_output_dir, exist_ok=True)
    mr_target_path = os.path.join(first_mr_output_dir, os.path.basename(mr_file))
    
    if os.path.isfile(mr_file):
        move_file(mr_file, mr_target_path)
    else:
        print(f"mr 파일을 찾을 수 없습니다... : {mr_file}")
        return None, None

    if not os.path.isfile(vocal_file):
        print(f"vocal 파일을 찾을 수 없습니다.. : {vocal_file}")
        return None, None

    print("UVR-MDX-NET-Voc_FT 모델 로드 및 분리 시작......")
    separator.load_model('UVR-MDX-NET-Voc_FT.onnx')
    output_file_paths_2 = separator.separate(vocal_file)
    print(f"UVR-MDX-NET-Voc_FT 모델로 분리 완료! 다음과 같이 분리 되었습니다. : {output_file_paths_2}")
    trash_file_list.extend(output_file_paths_2)

    print("UVR-De-Echo-Aggressive 모델 로드 및 분리 시작......")
    separator.load_model('UVR-De-Echo-Aggressive.pth')
    output_file_paths_3 = separator.separate(output_file_paths_2[1])
    print(f"UVR-De-Echo-Aggressive 모델로 분리 완료! 다음과 같이 분리 되었습니다. : {output_file_paths_3}")
    trash_file_list.append(output_file_paths_3[1])
    
    for file_path in trash_file_list:
        delete_file(file_path)

    elapsed_time = time.time() - start_time
    print(f"오디오 분리 완료! 걸린 시간 :  {elapsed_time:.2f}초")

    return output_file_paths_3[0], mr_target_path

def single_inference(input_audio):
    start_time = time.time()

    spk_id = config_data['inference']['spk_id']
    transform = config_data['inference']['transform']
    f0_file = config_data['inference']['f0_file']
    f0_method = config_data['inference']['f0_method']
    file_index = config_data['model']['file_index']
    index_rate = config_data['inference']['index_rate']
    filter_radius = config_data['inference']['filter_radius']
    resample_sr = config_data['inference']['resample_sr']
    rms_mix_rate = config_data['inference']['rms_mix_rate']
    protect = config_data['inference']['protect']
    output_info = config_data['paths']['output_info']
    output_audio = config_data['paths']['output_audio']

    print("단일 추론 시작...")
    try:
        os.makedirs(os.path.dirname(output_audio), exist_ok=True)

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
        print("단일 추론 완료!")
        
        elapsed_time = time.time() - start_time
        print(f"걸린 시간 :  {elapsed_time:.2f}초")
        
        return output_audio
    except Exception as e:
        print(f"단일 추론 중 에러가 발생했습니다... : {e}")
        print(traceback.format_exc())
        return None

def mix_audio(vocal_path, mr_path, output_path):
    start_time = time.time()

    try:
        vocal = AudioSegment.from_wav(vocal_path)
        mr = AudioSegment.from_wav(mr_path)

        if len(vocal) > len(mr):
            vocal = vocal[:len(mr)]
        else:
            mr = mr[:len(vocal)]

        mixed = vocal.overlay(mr)
        mixed.export(output_path, format='wav')
        print(f'vocal과 mr이 믹싱 완료! 다음 위치에 저장되었습니다! : {output_path}')
        
        elapsed_time = time.time() - start_time
        print(f"걸린 시간 : {elapsed_time:.2f}초 ")
    except Exception as e:
        print(f"오디오 믹싱 중 에러가 발생했습니다... : {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    total_start_time = time.time()
    
    input_audio_path = config_data['paths']['input_audio']
    first_mr_output_dir = config_data['paths']['first_mr_output_dir']
    output_path = config_data['paths']['output_audio']
    
    processed_audio_path, mr_path = audio_sep(input_audio_path, first_mr_output_dir)
    
    if processed_audio_path is not None and mr_path is not None:
        output_audio_path = single_inference(processed_audio_path)
        if output_audio_path is not None:
            mix_audio(output_audio_path, mr_path, output_path)
        else:
            print("RVC 추론에 실패했습니다. 오디오 믹싱이 수행되지 않습니다...")
    else:
        print("오디오 분리에 실패했습니다. RVC 추론이 수행되지 않습니다...")
    
    total_elapsed_time = time.time() - total_start_time
    print(f"총 프로세스 완료 시간 : {total_elapsed_time:.2f}초")
