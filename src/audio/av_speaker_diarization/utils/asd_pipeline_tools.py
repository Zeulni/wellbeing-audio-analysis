import torch
import os
import subprocess
import glob
import cv2
import sys
import time
import numpy
import pickle
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.VideoFileClip import AudioFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips

from src.audio.utils.constants import ASD_DIR
from src.audio.utils.constants import VIDEOS_DIR

class ASDPipelineTools:
    def __init__(self) -> None:
        self.logger = None
    
    def set_logger(self, logger) -> None:
        self.logger = logger
        
    def get_logger(self) -> None:
        return self.logger

    def write_to_terminal(self, text, argument = "") -> None:
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + " " + argument + "\r\n")
        
        if self.logger != None:
            self.logger.log(text + " " + argument)

    def safe_pickle_file(self, save_path, data, text = "pickle file stored", text_argument = "") -> None:
        with open(save_path, 'wb') as fil:
            pickle.dump(data, fil)
        self.write_to_terminal(text, text_argument)

    def get_device(self) -> str:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.write_to_terminal("Detected device (Cuda/CPU): ", str(device))
        
        return device

    def get_frames_per_second(self, video_path: str) -> int:
        video = cv2.VideoCapture(video_path)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        self.write_to_terminal("Frames per second: ", str(fps))
        video.release()

        return fps

    def get_num_total_frames(self, video_path: str) -> int:
        video = cv2.VideoCapture(video_path)
        num_total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.write_to_terminal("Total number of frames: ", str(num_total_frames))
        video.release()

        return num_total_frames
    
    def create_video_copy_25fps(self, video_path: str, video_name: str, save_path: str) -> None:
        # Load the video 
        clip = VideoFileClip(video_path)
        
        new_video_name = video_name + "_25fps"
        new_video_path = os.path.join(save_path, new_video_name + ".mp4")
        new_num_frames_per_sec = 25
        
        
        # If the new video was already created, then return
        if os.path.isfile(new_video_path):
            self.write_to_terminal("A copy of the video was already created with 25 fps: ", new_video_path)
            return new_video_path, new_video_name, new_num_frames_per_sec

        new_clip = clip.set_fps(new_num_frames_per_sec)
        
        new_clip.write_videofile(new_video_path)
        
        self.write_to_terminal("A copy of the video was created with 25 fps: ", new_video_path)

        return new_video_path, new_video_name, new_num_frames_per_sec

    def download_model(self, pretrain_model_path: str) -> None:
        path = os.path.join(ASD_DIR, pretrain_model_path)
        if os.path.isfile(path) == False: # Download the pretrained model
            Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
            cmd = "gdown --id %s -O %s"%(Link, path)
            subprocess.call(cmd, shell=True, stdout=None)
            
            
    def get_video_path(self, video_name) -> tuple:
        
        # video path is the absolute path from root to the video (e.g. .mp4)
        video_path = glob.glob(os.path.join(VIDEOS_DIR, video_name + '.*'))
        
        if not video_path:
            video_path = str(VIDEOS_DIR / video_name)
            raise Exception("No video found for path:  " + video_path)
        else:
            video_path = video_path[0]

        # video path is the absolute path to the folder where all the resulting files are located
        save_path = os.path.join(VIDEOS_DIR, video_name)

        return video_path, save_path

    def extract_video(self, pyavi_path, video_path, duration, n_data_loader_thread, start, num_frames_per_sec) -> str:
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
            self.write_to_terminal("Extract the video and save in ", extracted_video_path)

        return extracted_video_path

    def extract_audio_from_video(self, audio_storage_folder, video_path, n_data_loader_thread, video_name) -> str:
        audioFilePath = os.path.join(audio_storage_folder, video_name + '.wav')
        command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
            (video_path, n_data_loader_thread, audioFilePath))
        subprocess.call(command, shell=True, stdout=None)   
        
        self.write_to_terminal("Extract the audio and save in ", audioFilePath)
        
        return audioFilePath

    def visualization(self, tracks, scores, total_frames, video_path, pyavi_path, num_frames_per_sec, n_data_loader_thread, audio_file_path) -> None:
        
        video_only_path = os.path.join(pyavi_path, 'video_only.avi')
        video_out_path = os.path.join(pyavi_path, 'video_out.avi')
        
        # Check if videos are already created (if yes, then directly return)
        if os.path.isfile(video_only_path) and os.path.isfile(video_out_path):
            self.write_to_terminal("Videos already created, skip the visualization step")
            return
        
        # CPU: visulize the result for video format
        all_faces = [[] for i in range(total_frames)]
        
        # *Pick one track
        for tidx, track in enumerate(tracks):
            score = scores[tidx]
            # *Go through each frame in the selected track
            for fidx, frame in enumerate(track['track']['frame'].tolist()):
                s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
                s = numpy.mean(s)
                # *Store for each frame the bounding box and score (for each of the detected faces/tracks over time)
                all_faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})

        colorDict = {0: 0, 1: 255}
        # Get height and width in pixel of the video
        cap = cv2.VideoCapture(video_path)
        fw = int(cap.get(3))
        fh = int(cap.get(4))
        vOut = cv2.VideoWriter(os.path.join(pyavi_path, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), num_frames_per_sec, (fw,fh))

        # Instead of using the stored images in Pyframes, load the images from the video (which is stored at videoPath) and draw the bounding boxes there
        # CPU: visulize the result for video format
        # *Go through each frame
        for fidx in range(total_frames):
            # *Load the frame from the video
            cap = cv2.VideoCapture(video_path)
            cap.set(1, fidx)
            ret, image = cap.read()
            # *Within each frame go through each face and draw the bounding box
            for face in all_faces[fidx]:
                clr = colorDict[int((face['score'] >= 0))]
                txt = round(face['score'], 1)
                cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
                cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
                # Also add the track number as text (but below the bounding box)
                cv2.putText(image,'%s'%(face['track']), (int(face['x']-face['s']), int(face['y']+face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
            vOut.write(image)
        vOut.release()
        
        self.write_to_terminal("Visualizatin finished - now it will be saved.")

        command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
            (video_only_path, audio_file_path, \
            n_data_loader_thread, video_out_path)) 
        output = subprocess.call(command, shell=True, stdout=None)
        
        self.write_to_terminal("Visualization video saved to", os.path.join(pyavi_path,'video_out.avi'))

    def cut_track_videos(self, track_speaking_segments, pyavi_path, video_path, n_data_loader_thread) -> None:
        # Using the trackSpeakingSegments, extract for each track the video segments from the original video (with moviepy)
        # Concatenate all the different subclip per track into one video
        # Go through each track
        for tidx, track in enumerate(track_speaking_segments):
            # Check whether the track is empty
            if len(track) == 0:
                continue

            # Only create the video if the output file does not exist
            cutted_file_name = os.path.join(pyavi_path, 'track_%s.mp4' % (tidx))
            if os.path.exists(cutted_file_name):
                continue

            # Create the list of subclips
            clips = []
            for segment in track:
                clips.append(VideoFileClip(video_path).subclip(segment[0], segment[1]))
            # Concatenate the clips
            final_clip = concatenate_videoclips(clips)
            # Write the final video
            final_clip.write_videofile(cutted_file_name, threads=n_data_loader_thread)
            # final_clip.write_videofile(cutted_file_name, threads=n_data_loader_thread, logger=None)