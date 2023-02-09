import torch
import os
import subprocess
import glob
import cv2
import sys
import time

from src.audio.utils.constants import ASD_DIR

def get_device() -> str:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Detected device (Cuda/CPU): ", device)
    
    return device

def get_frames_per_second(video_path: str) -> int:
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print("Frames per second: ", fps)
    video.release()

    return fps

def get_num_total_frames(video_path: str) -> int:
    video = cv2.VideoCapture(video_path)
    num_total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total number of frames: ", num_total_frames)
    video.release()

    return num_total_frames

def download_model(pretrain_model_path: str) -> None:
    path = os.path.join(ASD_DIR, pretrain_model_path)
    if os.path.isfile(path) == False: # Download the pretrained model
        Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
        cmd = "gdown --id %s -O %s"%(Link, path)
        subprocess.call(cmd, shell=True, stdout=None)
        
        
def get_video_path(video_folder, video_name) -> tuple:
    
    video_folder_path = os.path.join(ASD_DIR, video_folder)
    
    # video path is the absolute path from root to the video (e.g. .mp4)
    video_path = glob.glob(os.path.join(video_folder_path, video_name + '.*'))[0]

    # video path is the absolute path to the folder where all the resulting files are located
    save_path = os.path.join(video_folder_path, video_name)

    return video_path, save_path

def extract_video(pyavi_path, video_path, duration, n_data_loader_thread, start, num_frames_per_sec) -> str:
    # Cut the video if necessary
    extracted_video_path = os.path.join(pyavi_path, 'video.avi')
    # If duration did not set, just use the provided video, otherwise extract the video from 'start' to 'start + duration'
    if duration == 0:
        extracted_video_path = video_path
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Video will not be cutted and remains in %s \r\n" %(extracted_video_path))
    else:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r %d %s -loglevel panic" % \
            (video_path, n_data_loader_thread, start, start + duration, num_frames_per_sec, extracted_video_path))
        subprocess.call(command, shell=True, stdout=None)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" %(extracted_video_path))

    return extracted_video_path

def extract_audio_from_video(pyavi_path, video_path, n_data_loader_thread) -> str:
    audioFilePath = os.path.join(pyavi_path, 'audio.wav')
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
        (video_path, n_data_loader_thread, audioFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(audioFilePath))
    
    return audioFilePath