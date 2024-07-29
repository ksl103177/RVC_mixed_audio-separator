import subprocess
import os
import shutil

def download_youtube_videos(urls):
    yt_dlp_path = shutil.which('yt-dlp')
    if yt_dlp_path is None:
        print("yt-dlp not found. Please ensure it is installed and available in your PATH.")
        return
    
    for url in urls:
        if not url:
            print("유효하지 않은 URL이 있습니다. 건너뛰겠습니다.")
            continue
        try:
            print(f"다운로드 중... : {url}")
            # video download
            # video_command = [yt_dlp_path, '-f', 'bestvideo', '--verbose', '--concurrent-fragments', '8', url]
            # audio download
            audio_command = [yt_dlp_path, '-f', 'bestaudio', '--extract-audio', '--audio-format', 'wav', '--verbose', '--concurrent-fragments', '8', url]
            
            # subprocess.run(video_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # print("영상 다운로드 완료...")
            
            subprocess.run(audio_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("오디오 다운로드 및 변환 완료...")
            
            print(f"{url} 다운로드 완료.")
        except subprocess.CalledProcessError as e:
            print(f"다운로드 실패... : {url} : {str(e)}")
        except Exception as e:
            print(f"An error occurred with {url}: {str(e)}")

video_urls = [
              '',
             ]

download_youtube_videos(video_urls)
