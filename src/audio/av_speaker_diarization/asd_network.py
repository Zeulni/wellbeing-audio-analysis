import os
import sys
import itertools
import tqdm
import python_speech_features
import torch
import numpy
import math
import multiprocessing as mp
from pydub import AudioSegment

from src.audio.av_speaker_diarization.talknet import talkNet


class ASDNetwork():
    def __init__(self, device, pretrain_model, num_frames_per_sec, frames_face_tracking, folder_frames_storage, total_frames, multiprocessing) -> None:
        self.device = device
        self.pretrain_model = pretrain_model
        self.num_frames_per_sec = num_frames_per_sec
        self.frames_face_tracking = frames_face_tracking
        self.total_frames = total_frames
        self.multiprocessing = multiprocessing
        
        self.faces_frames = None
        self.audio_file_path = None
        self.s = None 
        self.number_tracks = None
        
        self.folder_frames_storage = folder_frames_storage
        
    def talknet_network(self, all_tracks, audio_file_path, track_frame_overview) -> list:
        # GPU: active speaker detection by pretrained TalkNet
        s = talkNet(device=self.device).to(self.device)
        s.loadParameters(self.pretrain_model)
        sys.stderr.write("Model %s loaded from previous state! \r\n"%self.pretrain_model)
        s.eval()	
        
        self.audio_file_path = audio_file_path
        self.s = s
        
        self.number_tracks = len(all_tracks)

        if self.device.type == 'cuda' or self.multiprocessing == False:            
            # Multiprocessing for GPU did not work on Colab
            all_scores = []
            for tidx, track in enumerate(tqdm.tqdm(all_tracks)):
                track_scores = self.calculate_scores(tidx, track, track_frame_overview)
                all_scores.append(track_scores)        
        else:     
            # Using the "spawn" method to create a pool of worker processes and show progress with tqdm
            mp.freeze_support()
            with mp.get_context("spawn").Pool(processes=mp.cpu_count()) as pool:
                all_scores = list(tqdm.tqdm(pool.imap(self.calculate_scores_mp, zip(enumerate(all_tracks), itertools.repeat(track_frame_overview))), total=len(all_tracks)))
            
        return all_scores
    
    def calculate_scores_mp(self, args):
        track_idx_track, track_frame_overview = args
        track_idx, track = track_idx_track
        return self.calculate_scores(track_idx, track, track_frame_overview)
    
    def extract_audio(self, audio_file, track) -> tuple:
        audio_start  = (track['frame'][0]) / self.num_frames_per_sec
        audio_end    = (track['frame'][-1]+1) / self.num_frames_per_sec
        sound = AudioSegment.from_wav(audio_file)
        
        segment = sound[audio_start*1000:audio_end*1000]
        samplerate = segment.frame_rate
        trans_segment = numpy.array(segment.get_array_of_samples(), dtype=numpy.int16)
        
        # Audio Preparation (squeezing + stretching)
        length_segment = len(trans_segment)
        
        # Shorten the length of the audio segment by factor self.frames_face_tracking
        trans_segment = trans_segment[::self.frames_face_tracking]
        
        # Stretch it again to fill in the intermediate values with the same value
        trans_segment = numpy.repeat(trans_segment, self.frames_face_tracking)
        
        # Cut the audio segment to the same length as before (length_segment)
        trans_segment = trans_segment[:length_segment]
    

        return trans_segment, samplerate
    
    # def get_video_feature(self, tidx) -> numpy.ndarray:
         
    #     # Load the faces array from self.file_path_frames_storage using memmap, then extract only the relevant track into the memory
    #     length_frames = int(self.total_frames / self.frames_face_tracking)
    #     faces = numpy.memmap(self.file_path_frames_storage, mode="r+", shape=(self.number_tracks, length_frames, 112, 112), dtype=numpy.uint8)
    #     # track_data = faces[tidx]
        
    #     # Convert faces directly to a torch tensor, and put it on the GPU
    #     track_data = torch.tensor(faces[tidx]).to(self.device)

    #     # Change to float32
    #     track_data = track_data.to(torch.float32).to(self.device)
       
    #     return track_data
    
    def get_video_feature(self, tidx, track_frame_overview) -> numpy.ndarray:
        
        # Load the faces array for the relevant track using memmap
        length_frames = track_frame_overview[tidx]
        file_path = os.path.join(self.folder_frames_storage, f"frames_track_{tidx}.npz")
        faces = numpy.memmap(file_path, mode="r", shape=(length_frames, 112, 112), dtype=numpy.uint8)

        # Convert faces directly to a torch tensor, and put it on the GPU
        track_data = torch.tensor(faces).to(self.device)

        # Change to float32
        track_data = track_data.to(torch.float32).to(self.device)

        return track_data
    
    def calculate_scores(self, tidx, track, track_frame_overview) -> list:
        # duration_set = {1,2,4,6} # To make the result more reliable
        duration_set = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
        
        segment, samplerate = self.extract_audio(self.audio_file_path, track)
        audio_feature = python_speech_features.mfcc(segment, samplerate, numcep = 13, winlen = 0.025, winstep = 0.010)

        # Instead of saving the cropped the video, call the crop_track function to return the faces (without saving them)
        # * Problem: The model might have been trained with compressed image data (as I directly load them and don't save them as intermediate step, my images are slightly different)
        video_feature = self.get_video_feature(tidx, track_frame_overview)         

        # Reset the length of the input video to the original one (cut it to the original length, if one frame too much was added)
        video_feature = video_feature.repeat_interleave(self.frames_face_tracking, dim=0)
        video_feature = video_feature[:self.total_frames]     

        length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100, video_feature.shape[0])
        audio_feature = audio_feature[:int(round(length * 100)),:]
        video_feature = video_feature[:int(round(length * self.num_frames_per_sec)),:,:]
        track_scores = [] # Evaluation use TalkNet
        for duration in duration_set:
            batch_size = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batch_size):
                    inputA = torch.FloatTensor(audio_feature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).to(self.device)
                    inputV = video_feature[i * duration * self.num_frames_per_sec: (i+1) * duration * self.num_frames_per_sec,:,:].unsqueeze(0).to(self.device)
                    embedA = self.s.model.forward_audio_frontend(inputA).to(self.device)
                    embedV = self.s.model.forward_visual_frontend(inputV).to(self.device)
    
                    embedA, embedV = self.s.model.forward_cross_attention(embedA, embedV)
                    out = self.s.model.forward_audio_visual_backend(embedA, embedV)
                    score = self.s.lossAV.forward(out, labels = None)
                    scores.extend(score)
            track_scores.append(scores)
        track_scores = numpy.round((numpy.mean(numpy.array(track_scores), axis = 0)), 1).astype(float)
            
        return track_scores
    