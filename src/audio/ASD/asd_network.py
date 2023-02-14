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
    def __init__(self, device, pretrain_model, num_frames_per_sec, frames_face_tracking) -> None:
        self.device = device
        self.pretrain_model = pretrain_model
        self.num_frames_per_sec = num_frames_per_sec
        self.frames_face_tracking = frames_face_tracking
        
        self.faces_frames = None
        self.audio_file_path = None
        self.s = None 
        
    def talknet_network(self, all_tracks, faces_frames, audio_file_path) -> list:
        # GPU: active speaker detection by pretrained TalkNet
        s = talkNet(device=self.device).to(self.device)
        s.loadParameters(self.pretrain_model)
        sys.stderr.write("Model %s loaded from previous state! \r\n"%self.pretrain_model)
        s.eval()	
        
        self.faces_frames = faces_frames
        self.audio_file_path = audio_file_path
        self.s = s

        # allScores= []
        # for tidx, track in enumerate(all_tracks):

        #     allScore = self.calculate_scores(tidx, track)
        #     allScores.append(allScore)	
        
        
        # Create a pool of worker processes
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # Map the calculate_scores_mp function to the list of tracks
            allScores = list(pool.map(self.calculate_scores_mp, enumerate(all_tracks)))
            
        return allScores
    
    def calculate_scores_mp(self, tidx_track):
        tidx, track = tidx_track
        return self.calculate_scores(tidx, track)
    
    def extract_audio(self, audio_file, track) -> tuple:
        audioStart  = (track['frame'][0]) / self.num_frames_per_sec
        audioEnd    = (track['frame'][-1]+1) / self.num_frames_per_sec
        sound = AudioSegment.from_wav(audio_file)
        
        segment = sound[audioStart*1000:audioEnd*1000]
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
    
    
    def calculate_scores(self, tidx, track) -> list:
        # durationSet = {1,2,4,6} # To make the result more reliable
        durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
        
        segment, samplerate = self.extract_audio(self.audio_file_path, track)

        audioFeature = python_speech_features.mfcc(segment, samplerate, numcep = 13, winlen = 0.025, winstep = 0.010)

        # Instead of saving the cropped the video, call the crop_track function to return the faces (without saving them)
        # * Problem: The model might have been trained with compressed image data (as I directly load them and don't save them as intermediate step, my images are slightly different)
        videoFeature = self.faces_frames[tidx]
        # Remove all frames that have the value 0 (as they are not used)
        videoFeature = videoFeature[videoFeature.sum(axis=(1,2)) != 0]

        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
        audioFeature = audioFeature[:int(round(length * 100)),:]
        videoFeature = videoFeature[:int(round(length * self.num_frames_per_sec)),:,:]
        allScore = [] # Evaluation use TalkNet
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).to(self.device)
                    inputV = videoFeature[i * duration * self.num_frames_per_sec: (i+1) * duration * self.num_frames_per_sec,:,:].unsqueeze(0).to(self.device)
                    embedA = self.s.model.forward_audio_frontend(inputA).to(self.device)
                    embedV = self.s.model.forward_visual_frontend(inputV).to(self.device)
    
                    embedA, embedV = self.s.model.forward_cross_attention(embedA, embedV)
                    out = self.s.model.forward_audio_visual_backend(embedA, embedV)
                    score = self.s.lossAV.forward(out, labels = None)
                    scores.extend(score)
            allScore.append(scores)
        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)

        # TODO: Option 1
        # To compensate for the skipping of frames, repeat the score for each frame (so it has the same length again)
        allScore = numpy.repeat(allScore, self.frames_face_tracking)

        # To make sure the length is not longer than the video, crop it (if its the same length, just cut 3 frames off to be on the safe side)
        if allScore.shape[0] > track['bbox'].shape[0]:
            allScore = allScore[:track['bbox'].shape[0]]
        elif (allScore.shape[0] - track['bbox'].shape[0]) >= -3:
            allScore = allScore[:-3]
            
        return allScore
    