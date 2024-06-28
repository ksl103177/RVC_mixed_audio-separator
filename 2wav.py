import subprocess
import os

def convert_mp4_to_wav(input_file, output_file):
    command = [
        'ffmpeg', 
        '-i', input_file, 
        '-vn', 
        '-acodec', 'pcm_s16le', 
        '-ar', '44100',
        '-ac', '2',
        output_file
    ]
    subprocess.run(command, check=True)

def convert_all_mp4_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.mp3'):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename.replace('.mp3', '.wav'))
            convert_mp4_to_wav(input_file, output_file)
            # print(f'Converted {input_file} to {output_file}')

input_folder = 'C:/RVC/RVC_test/test/test_01'
output_folder = 'C:/RVC/RVC_test/test/test_02'

convert_all_mp4_in_folder(input_folder, output_folder)