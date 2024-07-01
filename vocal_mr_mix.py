from pydub import AudioSegment

vocal_path = ''
mr_path = ''

vocal = AudioSegment.from_wav(vocal_path)
mr = AudioSegment.from_wav(mr_path)

if len(vocal) > len(mr):
    vocal = vocal[:len(mr)]
else:
    mr = mr[:len(vocal)]

mixed = vocal.overlay(mr)

output_path = ''
mixed.export(output_path, format='wav')

print(f'vocal과 mr이 mix 되었습니다! -> {output_path}')
