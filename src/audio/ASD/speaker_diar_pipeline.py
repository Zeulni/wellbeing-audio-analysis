# Code adapted based on "TalkNet-ASD" by TaoRuijie
# Original repository: https://github.com/TaoRuijie/TalkNet-ASD
# Obtained on: 2023-01-31

import os
import pickle

# from shutil import rmtree

# Disabled scene detection for now, because cutted teamwork videos have no change in scene 
# if you want to add it later, have a look at the original repo: https://github.com/TaoRuijie/TalkNet-ASD

from src.audio.ASD.utils.asd_pipeline_tools import get_device
from src.audio.ASD.utils.asd_pipeline_tools import download_model
from src.audio.ASD.utils.asd_pipeline_tools import get_video_path
from src.audio.ASD.utils.asd_pipeline_tools import get_frames_per_second
from src.audio.ASD.utils.asd_pipeline_tools import extract_video
from src.audio.ASD.utils.asd_pipeline_tools import get_num_total_frames
from src.audio.ASD.utils.asd_pipeline_tools import extract_audio_from_video
from src.audio.ASD.utils.asd_pipeline_tools import safe_pickle_file
from src.audio.ASD.utils.asd_pipeline_tools import write_to_terminal
from src.audio.ASD.utils.asd_pipeline_tools import visualization

from src.audio.ASD.face_detection import FaceDetector
from src.audio.ASD.face_tracking import FaceTracker
from src.audio.ASD.asd_network import ASDNetwork
from src.audio.ASD.crop_tracks import CropTracks
from src.audio.ASD.speaker_diarization import SpeakerDiarization

from src.audio.utils.constants import ASD_DIR

class ASDPipeline:
	def __init__(self, args):
		self.video_name = args.get("VIDEO_NAME","001")
		self.video_folder = args.get("VIDEO_FOLDER","src/audio/ASD/demo")
		self.pretrain_model = args.get("PRETRAIN_ASD_MODEL","pretrain_TalkSet.model")
		self.pretrain_model = os.path.join(ASD_DIR, self.pretrain_model)
  
		self.n_data_loader_thread = args.get("N_DATA_LOADER_THREAD",32)
		self.face_det_scale = args.get("FACE_DET_SCALE",0.25)
		self.min_track = args.get("MIN_TRACK",10)
		self.num_failed_det = args.get("NUM_FAILED_DET",25)
		self.min_face_size = args.get("MIN_FACE_SIZE",1)
		self.crop_scale = args.get("CROP_SCALE",0.40)
		self.frames_face_tracking = args.get("FRAMES_FACE_TRACKING",1)
		self.start = args.get("START",0)
		self.duration = args.get("DURATION",0)
		self.threshold_same_person = args.get("THRESHOLD_SAME_PERSON",0.15)
		self.create_track_videos = args.get("CREATE_TRACK_VIDEOS",True)
		self.include_visualization = args.get("INCLUDE_VISUALIZATION",True)
  
		#warnings.filterwarnings("ignore")

		write_to_terminal("Only every xth frame will be analyzed for faster processing:", str(self.frames_face_tracking))

		# Check whether GPU is available on cuda (NVIDIA), then mps (Macbook), otherwise use cpu
		self.device = get_device()

		# Download the pretrained model if not exist
		download_model(self.pretrain_model)
	
		self.video_path, self.save_path = get_video_path(self.video_folder, self.video_name)

		# Initialization
		self.pyavi_path = os.path.join(self.save_path, 'pyavi')
		self.pywork_path = os.path.join(self.save_path, 'pywork')
	
		self.num_frames_per_sec = get_frames_per_second(self.video_path)
		self.total_frames = get_num_total_frames(self.video_path)
		
		# Extract the video from the start time to the duration
		self.video_path = extract_video(self.pyavi_path, self.video_path, self.duration, self.n_data_loader_thread, self.start, self.num_frames_per_sec)	
     
		# TODO: Check if that's nevertheless necessary based on UX
		# if os.path.exists(save_path):
		# 	rmtree(save_path)

		if not os.path.exists(self.pywork_path): # Save the results in this process by the pckl method
			os.makedirs(self.pywork_path)
	
		if not os.path.exists(self.pyavi_path): # The path for the input video, input audio, output video
			os.makedirs(self.pyavi_path) 
   
		# The pickle files
		self.faces = None
		self.tracks = None
		self.score = None
   
		# Initialize the face detector
		self.face_detector = FaceDetector(self.device, self.video_path, self.frames_face_tracking, self.face_det_scale, self.pywork_path)
  
		# Initialize the face tracker
		self.face_tracker = FaceTracker(self.num_failed_det, self.min_track, self.min_face_size)
  
		# Initialize the track cropper
		self.track_cropper = CropTracks(self.video_path, self.total_frames, self.frames_face_tracking, self.crop_scale, self.device)
  
		# Initialize the ASD network
		self.asd_network = ASDNetwork(self.device, self.pretrain_model, self.num_frames_per_sec)
  
		# Initialize the speaker diarization
		self.speaker_diarization = SpeakerDiarization(self.pyavi_path, self.video_path, self.video_name, self.n_data_loader_thread, 
                                                	  self.threshold_same_person, self.create_track_videos, self.total_frames, self.num_frames_per_sec)
	
  
	def __check_asd_done(self) -> bool:
		# If pickle files exist in the pywork folder, then directly load the scores and tracks pickle files
		if os.path.exists(os.path.join(self.pywork_path, 'scores.pkl')) and os.path.exists(os.path.join(self.pywork_path, 'tracks.pkl')):
			with open(os.path.join(self.pywork_path, 'scores.pkl'), 'rb') as f:
				self.scores = pickle.load(f)
			with open(os.path.join(self.pywork_path, 'tracks.pkl'), 'rb') as f:
				self.tracks = pickle.load(f)
			write_to_terminal("ASD is done, scores and tracks are loaded from the pickle files.")
			return True
		return False

	def __check_face_detection_done(self) -> bool:
		if os.path.exists(os.path.join(self.pywork_path, 'faces.pckl')):
			with open(os.path.join(self.pywork_path, 'faces.pckl'), 'rb') as f:
				self.faces = pickle.load(f)
				write_to_terminal("Face detection is done, faces are loaded from the pickle files.")
				return True
		else:
			self.faces = None
			return False


	# Pipeline for the ASD algorithm
	def run(self) -> None:
		# This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
		# ```
		# .
		# ├── pyavi
		# │   ├── audio.wav (Audio from input video)
		# │   ├── video.avi (Copy of the input video, if duration is adjusted)
		# │   ├── video_only.avi (Output video without audio)
		# │   └── video_out.avi  (Output video with audio)
		# └── pywork
		#     ├── faces.pckl (face detection result)
		#     ├── scores.pckl (ASD result) - score values over time whether one speaks, for each detected face (video)
		#     └── tracks.pckl (face tracking result) - face bounding boxes over time for each detected face (video), per track x frames (e.g. in sample 7 tracks each ~500 frames)
		# ```
	
		# Checkpoint ASD (Assumption: If pickle files in pywork folder exist, ASD is done and all the other files exist (to re-run ASD delete pickle files))
		asd_done = self._ASDPipeline__check_asd_done()
		if asd_done == False:
	
			# Extract audio from video
			audio_file_path = extract_audio_from_video(self.pyavi_path, self.video_path, self.n_data_loader_thread)
		
			# Face detection (check for checkpoint first)
			face_detection_done = self._ASDPipeline__check_face_detection_done()
			if face_detection_done == False:
				self.faces = self.face_detector.s3fd_face_detection()
		
			# Face tracking
			all_tracks = []
			all_tracks.extend(self.face_tracker.track_shot_face_tracker(self.faces))
			write_to_terminal("Face tracks created - detected", str(len(all_tracks)))

			# Crop all the tracks from the video (are stored in CPU memory)
			self.tracks, faces_all_tracks = self.track_cropper.crop_tracks_from_videos_parallel(all_tracks)
			safe_pickle_file(self.pywork_path, "tracks.pkl", self.tracks, "Track saved in", self.pywork_path)

			# Active Speaker Detection by TalkNet
			self.scores = self.asd_network.talknet_network(all_tracks, faces_all_tracks, audio_file_path)
			safe_pickle_file(self.pywork_path, "scores.pkl", self.scores, "Scores extracted and saved in", self.pywork_path)
		

		# Visualization, save the result as the new video	
		if self.include_visualization == True:
			visualization(self.tracks, self.scores, self.total_frames, self.video_path, self.pyavi_path, self.num_frames_per_sec, self.n_data_loader_thread)
		
		# Speaker diarization
		self.speaker_diarization.run(self.tracks, self.scores)