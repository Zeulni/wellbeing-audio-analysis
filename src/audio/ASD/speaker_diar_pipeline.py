# Code adapted based on "TalkNet-ASD" by TaoRuijie
# Original repository: https://github.com/TaoRuijie/TalkNet-ASD
# Obtained on: 2023-01-31

import copy
import os
import pickle
import time

# from shutil import rmtree

# Disabled scene detection for now, because cutted teamwork videos have no change in scene 
# if you want to add it later, have a look at the original repo: https://github.com/TaoRuijie/TalkNet-ASD

from src.audio.ASD.face_detection import FaceDetector
from src.audio.ASD.face_tracking import FaceTracker
from src.audio.ASD.asd_network import ASDNetwork
from src.audio.ASD.crop_tracks import CropTracks
from src.audio.ASD.speaker_diarization import SpeakerDiarization

from src.audio.utils.constants import ASD_DIR

class ASDSpeakerDirPipeline:
	def __init__(self, args, num_frames_per_sec, total_frames, audio_file_path, video_path, save_path, video_name, asd_pipeline_tools, faces_id_path):
		self.video_name = video_name
		self.pretrain_model = args.get("PRETRAIN_ASD_MODEL","pretrain_TalkSet.model")
		self.pretrain_model = os.path.join(ASD_DIR, self.pretrain_model)
  
		self.n_data_loader_thread = args.get("N_DATA_LOADER_THREAD",32)
		self.face_det_scale = args.get("FACE_DET_SCALE",0.25)
		self.min_track = args.get("MIN_TRACK",25)
		self.num_failed_det = args.get("NUM_FAILED_DET",100)
		self.min_face_size = args.get("MIN_FACE_SIZE",1)
		self.crop_scale = args.get("CROP_SCALE",0.40)
		self.frames_face_tracking = args.get("FRAMES_FACE_TRACKING",2)
		self.start = args.get("START",0)
		self.duration = args.get("DURATION",0)
		self.threshold_same_person = args.get("THRESHOLD_SAME_PERSON",1.05)
		self.create_track_videos = args.get("CREATE_TRACK_VIDEOS",True)
		self.include_visualization = args.get("INCLUDE_VISUALIZATION",True)
		self.n_embeddings = args.get("N_EMBEDDINGS",10)
  
		self.asd_pipeline_tools = asd_pipeline_tools
		self.logger = self.asd_pipeline_tools.get_logger()
		self.asd_pipeline_tools.write_to_terminal("Only every xth frame will be analyzed for faster processing:", str(self.frames_face_tracking))

		# Check whether GPU is available on cuda (NVIDIA), then mps (Macbook), otherwise use cpu
		self.device = self.asd_pipeline_tools.get_device()

		# Download the pretrained model if not exist
		self.asd_pipeline_tools.download_model(self.pretrain_model)
	
		self.audio_file_path = audio_file_path
		self.video_path = video_path
		self.save_path = save_path

		# Initialization
		self.pyavi_path = os.path.join(self.save_path, 'pyavi')
		self.pywork_path = os.path.join(self.save_path, 'pywork')
		self.faces_id_path = faces_id_path
		self.tracks_faces_clustering_path = os.path.join(self.save_path, 'tracks_faces_clustering')
  
		self.file_path_faces_bbox = os.path.join(self.pywork_path,  "faces_bbox.pickle")
		self.file_path_scores = os.path.join(self.pywork_path,  "scores.pickle")
		self.file_path_tracks = os.path.join(self.pywork_path,  "tracks.pickle")
		self.file_path_track_frame_overview = os.path.join(self.pywork_path,  "track_frame_overview.pickle")
	
		self.num_frames_per_sec = num_frames_per_sec
		self.total_frames = total_frames
		
		# Extract the video from the start time to the duration
		self.video_path = self.asd_pipeline_tools.extract_video(self.pyavi_path, self.video_path, self.duration, self.n_data_loader_thread, self.start, self.num_frames_per_sec)	
     
		# TODO: Check if that's nevertheless necessary based on UX
		# if os.path.exists(save_path):
		# 	rmtree(save_path)

		if not os.path.exists(self.pywork_path): # Save the results in this process by the pckl method
			os.makedirs(self.pywork_path)
	
		if not os.path.exists(self.pyavi_path): # The path for the input video, input audio, output video
			os.makedirs(self.pyavi_path) 
   
		if not os.path.exists(self.faces_id_path): # Here are the found faces and their IDs stored
			os.makedirs(self.faces_id_path) 
   
		if not os.path.exists(self.tracks_faces_clustering_path):
			os.makedirs(self.tracks_faces_clustering_path)
   
		# The pickle files
		self.faces_bbox = None
		self.tracks = None
		self.faces_frames= None
		self.score = None
		self.track_frame_overview = None
   
		# Initialize the face detector
		self.face_detector = FaceDetector(self.device, self.video_path, self.frames_face_tracking, self.face_det_scale, self.pywork_path, self.total_frames)
  
		# Initialize the face tracker
		self.face_tracker = FaceTracker(self.num_failed_det, self.min_track, self.min_face_size)
  
		# Initialize the track cropper
		self.track_cropper = CropTracks(self.video_path, self.total_frames, self.frames_face_tracking, self.crop_scale, self.asd_pipeline_tools)
  
		# Initialize the ASD network
		self.asd_network = ASDNetwork(self.device, self.pretrain_model, self.num_frames_per_sec, self.frames_face_tracking, self.pywork_path, self.total_frames)
  
		# Initialize the speaker diarization
		self.speaker_diarization = SpeakerDiarization(self.pyavi_path, self.video_path, self.video_name, self.n_data_loader_thread, self.threshold_same_person, 
                                                	  self.create_track_videos, self.total_frames, self.num_frames_per_sec, self.save_path, self.faces_id_path, 
                                                   self.tracks_faces_clustering_path, self.crop_scale, self.asd_pipeline_tools, self.n_embeddings)
	

	def __check_face_detection_done(self) -> bool:
		if os.path.exists(self.file_path_faces_bbox):
			with open(self.file_path_faces_bbox, 'rb') as f:
				self.faces_bbox = pickle.load(f)
				self.asd_pipeline_tools.write_to_terminal("Face detection is done, faces are loaded from the pickle files.")
				return True
		else:
			self.faces_bbox = None
			return False

	def __check_face_cropping_done(self) -> bool:
		# Only check here for the tracks file, not for every single npz file
		if os.path.exists(self.file_path_tracks) and os.path.exists(self.file_path_track_frame_overview):
			with open(self.file_path_tracks, 'rb') as f:
				self.tracks = pickle.load(f)
			with open(self.file_path_track_frame_overview, 'rb') as f:
				self.track_frame_overview = pickle.load(f)
			self.asd_pipeline_tools.write_to_terminal("Face cropping is done, tracks are loaded from the pickle files.")
			return True
		else:
			self.tracks = None
			return False

	def __check_asd_done(self) -> bool:
		if os.path.exists(self.file_path_scores):
			with open(self.file_path_scores, 'rb') as f:
				self.scores = pickle.load(f)
				self.asd_pipeline_tools.write_to_terminal("ASD is done, scores are loaded from the pickle files.")
				return True
		else:
			self.scores = None
			return False

	def __check_pipeline_done(self) -> bool:
		# If pickle files exist in the pywork folder, then directly load the scores and tracks pickle files
		if os.path.exists(self.file_path_scores) and os.path.exists(self.file_path_tracks):
			with open(self.file_path_scores, 'rb') as f:
				self.scores = pickle.load(f)
			with open(self.file_path_tracks, 'rb') as f:
				self.tracks = pickle.load(f)
			self.asd_pipeline_tools.write_to_terminal("ASD is done, scores and tracks are loaded from the pickle files.")
			return True
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
		#     ├── faces_bbox.pckl (face detection result)
		#     ├── scores.pckl (ASD result) - score values over time whether one speaks, for each detected face (video)
		#     └── tracks.pckl (face tracking result) - face bounding boxes over time for each detected face (video), per track x frames (e.g. in sample 7 tracks each ~500 frames)
		# ```
 
		# Checkpoint ASD (Assumption: If pickle files in pywork folder exist, ASD is done and all the other files exist (to re-run ASD delete pickle files))
		pipeline_done = self._ASDSpeakerDirPipeline__check_pipeline_done()
		if pipeline_done == False:
  
			# Face detection (check for checkpoint first)
			start_time = time.perf_counter()
			face_detection_done = self._ASDSpeakerDirPipeline__check_face_detection_done()
			if face_detection_done == False:
				self.faces_bbox = self.face_detector.s3fd_face_detection()
				self.asd_pipeline_tools.safe_pickle_file(self.file_path_faces_bbox, self.faces_bbox, "Faces detected and stored in", self.pywork_path)
			end_time = time.perf_counter()
			print(f"--- Face detection done in {end_time - start_time:0.4f} seconds")
			self.logger.log("Face detection done in " + str(int(end_time - start_time)) + " seconds")
		
			# Face tracking
			start_time = time.perf_counter()
			all_tracks = []
			copy_faces_bbox = copy.deepcopy(self.faces_bbox)
			all_tracks.extend(self.face_tracker.track_shot_face_tracker(copy_faces_bbox))
			self.asd_pipeline_tools.write_to_terminal("Face tracks created - detected", str(len(all_tracks)))
			end_time = time.perf_counter()
			print(f"--- Face tracking done in {end_time - start_time:0.4f} seconds")
			self.logger.log("Face tracking done in " + str(int(end_time - start_time)) + " seconds")

			# Crop all the tracks from the video (are stored in CPU memory)
			start_time = time.perf_counter()
			face_cropping_done = self._ASDSpeakerDirPipeline__check_face_cropping_done()
			if face_cropping_done == False:
				# self.tracks, self.faces_frames = self.track_cropper.crop_tracks_from_videos_parallel(all_tracks)
				self.tracks, self.track_frame_overview = self.track_cropper.crop_tracks_from_videos_parallel(all_tracks, self.pywork_path)
				self.asd_pipeline_tools.safe_pickle_file(self.file_path_tracks, self.tracks, "Track saved in", self.pywork_path)
                # Save track_frame_overview as a pickle file
				self.asd_pipeline_tools.safe_pickle_file(self.file_path_track_frame_overview, self.track_frame_overview, "Track frame overview saved in", self.pywork_path)
			end_time = time.perf_counter()
			print(f"--- Track cropping done in {end_time - start_time:0.4f} seconds")
			self.logger.log("Track cropping done in " + str(int(end_time - start_time)) + " seconds")

			# Active Speaker Detection by TalkNet
			start_time = time.perf_counter()
			asd_done = self._ASDSpeakerDirPipeline__check_asd_done()
			if asd_done == False:
				self.scores = self.asd_network.talknet_network(all_tracks, self.audio_file_path, self.track_frame_overview)
				self.asd_pipeline_tools.safe_pickle_file(self.file_path_scores, self.scores, "Scores extracted and saved in", self.pywork_path)
			end_time = time.perf_counter()
			print(f"--- ASD done in {end_time - start_time:0.4f} seconds")
			self.logger.log("ASD done in " + str(int(end_time - start_time)) + " seconds")
		

		# Visualization, save the result as the new video	
		if self.include_visualization == True:
			start_time = time.perf_counter()
			self.asd_pipeline_tools.visualization(self.tracks, self.scores, self.total_frames, self.video_path, self.pyavi_path, self.num_frames_per_sec, self.n_data_loader_thread, self.audio_file_path)
			end_time = time.perf_counter()
			print(f"--- Visualization done in {end_time - start_time:0.4f} seconds")
			self.logger.log("Visualization done in " + str(int(end_time - start_time)) + " seconds")
  
		# Speaker diarization
		self.speaker_diarization.run(self.tracks, self.scores)
  
		return