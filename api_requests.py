import requests
import os
import shutil

def process_audio(file_path, local_output_dir, server_url, model_index):
    print("파일 업로드 및 처리 중...")

    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'model_index': model_index}
        response = requests.post(f"{server_url}/rvc_project/", files=files, data=data)

    print("서버 응답 대기 중...")

    print('Status code:', response.status_code)
    print('Response JSON:', response.json())

    response_data = response.json()
    if response.status_code == 200 and 'output_path' in response_data:
        output_path = response_data['output_path']
        filename = os.path.basename(output_path)

        print("파일 다운로드 중...")

        download_url = f"{server_url}/infer_output/{filename}"
        download_response = requests.get(download_url, stream=True)

        local_output_path = os.path.join(local_output_dir, filename)

        if download_response.status_code == 200:
            with open(local_output_path, 'wb') as local_file:
                download_response.raw.decode_content = True
                shutil.copyfileobj(download_response.raw, local_file)
            print(f'파일이 로컬에 저장되었습니다: {local_output_path}')
        else:
            print('파일 다운로드에 실패했습니다.')
    else:
        print('응답에 파일 경로가 포함되어 있지 않거나 처리에 실패했습니다.')

file_path = "C:/RVC/RVC_test/infer_song/local_infer_song.wav"  # 업로드할 파일 경로
local_output_dir = 'C:/RVC/RVC_test/infer_output'  # 최종적으로 만들어진 노래를 저장할 폴더 경로
server_url = 'http://114.110.130.187:20088'
model_index = 0  # 사용할 모델의 인덱스

process_audio(file_path, local_output_dir, server_url, model_index)
