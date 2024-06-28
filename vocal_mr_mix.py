from pydub import AudioSegment

# 보컬 파일과 엠알 파일의 경로를 지정합니다.
vocal_path = '/data/kdg_workspace/RVC_test/infer_output/output_infer_03.wav'
mr_path = '/data/kdg_workspace/RVC_test/infer_song/2_a song of hope_(Instrumental).wav'

# AudioSegment를 사용하여 오디오 파일을 불러옵니다.
vocal = AudioSegment.from_wav(vocal_path)
mr = AudioSegment.from_wav(mr_path)

# 두 오디오 파일의 길이를 맞춥니다.
if len(vocal) > len(mr):
    vocal = vocal[:len(mr)]
else:
    mr = mr[:len(vocal)]

# 보컬과 엠알을 섞습니다. (믹싱 비율을 조정할 수 있습니다. 예: vocal - 6dB, mr - 0dB)
mixed = vocal.overlay(mr)

# 결과를 새로운 파일로 저장합니다.
output_path = '/data/kdg_workspace/RVC_test/infer_output/mix_03.wav'
mixed.export(output_path, format='wav')

print(f'vocal과 mr이 mix 되었습니다. {output_path}')
