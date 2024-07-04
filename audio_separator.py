import os
import time
import shutil
import traceback
from pydub import AudioSegment
from audio_separator.separator import Separator

def move_file(file_path, target_path):
    try:
        if os.path.isfile(file_path):
            shutil.move(file_path, target_path)
            print(f"파일 '{file_path}'이'{target_path}'로 이동되었습니다.")
        else:
            print(f"해당 파일이 없습니다: '{file_path}'")
    except Exception as e:
        print(f"파일 '{file_path}'을'{target_path}'로 이동하는 중 오류 발생: {e}")

def delete_file(file_path):
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"파일 '{file_path}'이 삭제되었습니다.")
        else:
            print(f"해당 파일이 없습니다: '{file_path}'")
    except Exception as e:
        print(f"파일 '{file_path}'을 삭제하는 중 오류 발생: {e}")

def audio_sep(input_audio_path, first_mr_output_dir):
    start_time = time.time()
    
    input_extension = os.path.splitext(input_audio_path)[1].lower()
    input_base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
    supported_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.mp4']
    
    if input_extension not in supported_extensions:
        print(f"지원되지 않는 파일 형식: {input_extension}")
        return None, None
    
    if input_extension != '.wav':
        audio = AudioSegment.from_file(input_audio_path)
        input_audio_path = input_audio_path.replace(input_extension, '.wav')
        audio.export(input_audio_path, format='wav')
    
    separator = Separator()
    trash_file_list = []

    print("UVR_MDXNET_Main 모델로 로딩 및 분리 중...")
    separator.load_model('UVR_MDXNET_Main.onnx')
    output_file_paths_1 = separator.separate(input_audio_path)
    print(f"UVR_MDXNET_Main 모델로 분리 완료! 파일들: {output_file_paths_1}")
    
    mr_file = output_file_paths_1[0]
    vocal_file = output_file_paths_1[1]
    trash_file_list.extend(output_file_paths_1)

    if not os.path.isfile(vocal_file):
        print(f"보컬 파일을 찾을 수 없음: {vocal_file}")
        return None, None

    print("UVR-MDX-NET-Voc_FT 모델로 로딩 및 분리 중...")
    separator.load_model('UVR-MDX-NET-Voc_FT.onnx')
    output_file_paths_2 = separator.separate(vocal_file)
    print(f"UVR-MDX-NET-Voc_FT 모델로 분리 완료! 파일들: {output_file_paths_2}")
    trash_file_list.extend(output_file_paths_2)

    print("UVR-De-Echo-Aggressive 모델로 로딩 및 분리 중...")
    separator.load_model('UVR-De-Echo-Aggressive.pth')
    output_file_paths_3 = separator.separate(output_file_paths_2[1])
    print(f"UVR-De-Echo-Aggressive 모델로 분리 완료! 파일들: {output_file_paths_3}")
    trash_file_list.extend(output_file_paths_3)

    final_output_name = f"{input_base_name}_output.wav"
    final_output_path = os.path.join(first_mr_output_dir, final_output_name)
    move_file(output_file_paths_3[0], final_output_path)

    for file_path in set(trash_file_list) - {final_output_path}:
        delete_file(file_path)

    elapsed_time = time.time() - start_time
    print(f"오디오 분리 완료! 소요 시간: {elapsed_time:.2f} 초")

    return final_output_path, None

if __name__ == "__main__":
    input_audio_path = ''
    first_mr_output_dir = ''
    processed_audio_path, mr_path = audio_sep(input_audio_path, first_mr_output_dir)
    
    if processed_audio_path:
        print(f"처리된 오디오 경로: {processed_audio_path}")
    else:
        print("오디오 분리에 실패했습니다.")
