import sys
import tqdm
import python_speech_features
import torch
import numpy
import math
import multiprocessing as mp
from pydub import AudioSegment

from src.audio.ASD.talkNet import talkNet


class ASDNetwork():
    def __init__(self, device, pretrain_model, num_frames_per_sec, frames_face_tracking, file_path_frames_storage, total_frames) -> None:
        self.device = device
        self.pretrain_model = pretrain_model
        self.num_frames_per_sec = num_frames_per_sec
        self.frames_face_tracking = frames_face_tracking
        self.total_frames = total_frames
        
        self.faces_frames = None
        self.audio_file_path = None
        self.s = None 
        self.number_tracks = None
        
        self.file_path_frames_storage = file_path_frames_storage
        
    def talknet_network(self, all_tracks, faces_frames, audio_file_path) -> list:
        # GPU: active speaker detection by pretrained TalkNet
        s = talkNet(device=self.device).to(self.device)
        s.loadParameters(self.pretrain_model)
        sys.stderr.write("Model %s loaded from previous state! \r\n"%self.pretrain_model)
        s.eval()	
        
        self.faces_frames = faces_frames
        self.audio_file_path = audio_file_path
        self.s = s
        
        self.number_tracks = len(all_tracks)

        # all_scores= []
        # for tidx, track in enumerate(all_tracks):

        #     track_scores = self.calculate_scores(tidx, track)
        #     all_scores.append(track_scores)	
        
        
        # # Create a pool of worker processes
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # Map the calculate_scores_mp function to the list of tracks
            all_scores = list(pool.map(self.calculate_scores_mp, enumerate(all_tracks)))
            
        return all_scores
    
    def calculate_scores_mp(self, tidx_track):
        tidx, track = tidx_track
        return self.calculate_scores(tidx, track)
    
    def extract_audio(self, audio_file, track) -> tuple:
        audio_start  = (track['frame'][0]) / self.num_frames_per_sec
        audio_end    = (track['frame'][-1]+1) / self.num_frames_per_sec
        sound = AudioSegment.from_wav(audio_file)
        
        segment = sound[audio_start*1000:audio_end*1000]
        samplerate = segment.frame_rate
        trans_segment = numpy.array(segment.get_array_of_samples(), dtype=numpy.int16)

        # TODO: Option 1
        # For every 10th value ins trans_segment leave the value, for the rest set it to 0 (to compensate for video skipping frames)
        trans_segment_filtered = numpy.zeros(0)
        for i, value in enumerate(trans_segment):
        	if i % self.frames_face_tracking == 0:
        		trans_segment_filtered = numpy.append(trans_segment_filtered, value)

        # TODO: Option 1
        return trans_segment_filtered, samplerate
    
    def get_video_feature(self, tidx) -> numpy.ndarray:
        # Get the frames for the corresponding track from the frames_tracks.npz file in the pywork folder and then return it
    #    all_track_data = numpy.load(self.file_path_frames_storage, allow_pickle=True)
    #    all_track_data = all_track_data['faces']
    #    track_data = all_track_data[tidx]
    
         # faces = numpy.memmap(self.file_path_frames_storage, mode="r+", shape=(7, 501, 112, 112), dtype=float)
         
         # Load the faces array from self.file_path_frames_storage using memmap, then extract only the relevant track into the memory
            # and then return it
        faces = numpy.memmap(self.file_path_frames_storage, mode="r+", shape=(self.number_tracks, self.total_frames, 112, 112), dtype=float)
        track_data = faces[tidx]
        
        # Change it to float32
        track_data = track_data.astype(numpy.float32)

       
        return torch.tensor(track_data)
    
    def calculate_scores(self, tidx, track) -> list:
        # duration_set = {1,2,4,6} # To make the result more reliable
        duration_set = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
        
        segment, samplerate = self.extract_audio(self.audio_file_path, track)

        audio_feature = python_speech_features.mfcc(segment, samplerate, numcep = 13, winlen = 0.025, winstep = 0.010)

        # Instead of saving the cropped the video, call the crop_track function to return the faces (without saving them)
        # * Problem: The model might have been trained with compressed image data (as I directly load them and don't save them as intermediate step, my images are slightly different)
        # video_feature_old = self.faces_frames[tidx]
        video_feature = self.get_video_feature(tidx)
        
        # Check whether video_feature and video_feature_old are the same
        # if not numpy.array_equal(video_feature_old, video_feature):
        #     print("video_feature and video_feature_old are not the same")
        
        
        # Remove all frames that have the value 0 (as they are not used)
        video_feature = video_feature[video_feature.sum(axis=(1,2)) != 0]

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

        # TODO: Option 1
        # To compensate for the skipping of frames, repeat the score for each frame (so it has the same length again)
        track_scores = numpy.repeat(track_scores, self.frames_face_tracking)

        # To make sure the length is not longer than the video, crop it (if its the same length, just cut 3 frames off to be on the safe side)
        if track_scores.shape[0] > track['bbox'].shape[0]:
            track_scores = track_scores[:track['bbox'].shape[0]]
        elif (track_scores.shape[0] - track['bbox'].shape[0]) >= -3:
            track_scores = track_scores[:-3]
            
        return track_scores
    