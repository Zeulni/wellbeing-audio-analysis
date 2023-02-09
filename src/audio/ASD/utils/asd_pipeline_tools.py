import torch
import os
import subprocess
import glob
import cv2
import sys
import time
import numpy
import torchvision.transforms as transforms
from scipy import signal


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

def crop_tracks_from_videos_parallel(tracks, video_path, total_frames, frames_face_tracking, cs, device) -> tuple:
	# Instead of going only through one track in crop_track_faster, we only read the video ones and go through all the tracks
    # TODO: Maybe still needed if I make smooth transition between frames (instead of just fixing the bbox for 10 frames)
    # dets = {'x':[], 'y':[], 's':[]}
    # for det in track['bbox']: # Read the tracks
    # 	dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
    # 	dets['y'].append((det[1]+det[3])/2) # crop center x 
    # 	dets['x'].append((det[0]+det[2])/2) # crop center y


    # Go through all the tracks and get the dets values (not through the frames yet, only dets)
    # Save for each track the dats in a list
    dets = []
    for track in tracks:
        dets.append({'x':[], 'y':[], 's':[]})
        for fidx, det in enumerate(track['bbox']):
            if fidx%frames_face_tracking == 0:
                dets[-1]['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
                dets[-1]['y'].append((det[1]+det[3])/2)
                dets[-1]['x'].append((det[0]+det[2])/2)
            else:
                dets[-1]['s'].append(dets[-1]['s'][-1])
                dets[-1]['y'].append(dets[-1]['y'][-1])
                dets[-1]['x'].append(dets[-1]['x'][-1])
    
    # Go through all the tracks and smooth the dets values
    for track in dets:	
        track['s'] = signal.medfilt(track['s'], kernel_size=13)
        track['x'] = signal.medfilt(track['x'], kernel_size=13)
        track['y'] = signal.medfilt(track['y'], kernel_size=13)
    
    # Open the video
    vIn = cv2.VideoCapture(video_path)
    num_frames = total_frames

    # Create an empty array for the faces (num_tracks, num_frames, 112, 112)
    all_faces = torch.zeros((len(tracks), num_frames, 112, 112), dtype=torch.float32)

    # Define transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
        transforms.Lambda(lambda x: x.type(torch.uint8))
    ])
    
    # Loop over every frame, read the frame, then loop over all the tracks per frame and if available, crop the face
    for fidx in range(num_frames):
        vIn.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, image = vIn.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for tidx, track in enumerate(tracks):
            # In the current frame, first check whether the track has a bbox for this frame (if yes, perform opererations)
            if fidx in track['frame']:
                # Get the index of the frame in the track
                index = numpy.where(track['frame'] == fidx)
                index = int(index[0][0])
    
                # Calculate the bsi and pad the image
                bsi = int(dets[tidx]['s'][index] * (1 + 2 * cs))
                frame_image = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))

                bs  = dets[tidx]['s'][index]
                my  = dets[tidx]['y'][index] + bsi
                mx  = dets[tidx]['x'][index] + bsi

                # Crop the face from the image (depending on the track choose the image)
                face = frame_image[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]

                # Apply the transformations
                face = transform(face)

                # Store in the faces array
                all_faces[tidx, fidx, :, :] = face[0, :, :]

    
    # Close the video
    vIn.release()

    all_faces = all_faces.to(device)
    
    # Return the dets and the faces

    # Create a list where each element has the format {'track':track, 'proc_track':dets}
    proc_tracks = []
    for i in range(len(tracks)):
        proc_tracks.append({'track':tracks[i], 'proc_track':dets[i]})

    return proc_tracks, all_faces