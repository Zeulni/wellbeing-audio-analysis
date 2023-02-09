# Code adapted based on "TalkNet-ASD" by TaoRuijie
# Original repository: https://github.com/TaoRuijie/TalkNet-ASD
# Obtained on: 2023-01-31

import sys, time, os, subprocess, warnings, cv2, pickle, numpy, math, python_speech_features
# import multiprocessing

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips


from shutil import rmtree
from scipy.io import wavfile
from sklearn.metrics import accuracy_score, f1_score

# Disabled scene detection for now, because cutted teamwork videos have no change in scene 
# if you want to add it later, have a look at the original repo: https://github.com/TaoRuijie/TalkNet-ASD

from src.audio.ASD.utils.asd_pipeline_tools import get_device
from src.audio.ASD.utils.asd_pipeline_tools import download_model
from src.audio.ASD.utils.asd_pipeline_tools import get_video_path
from src.audio.ASD.utils.asd_pipeline_tools import get_frames_per_second
from src.audio.ASD.utils.asd_pipeline_tools import extract_video
from src.audio.ASD.utils.asd_pipeline_tools import get_num_total_frames
from src.audio.ASD.utils.asd_pipeline_tools import extract_audio_from_video
from src.audio.ASD.utils.asd_pipeline_tools import crop_tracks_from_videos_parallel

from src.audio.ASD.face_detection import FaceDetector
from src.audio.ASD.face_tracking import FaceTracker
from src.audio.ASD.asd_network import ASDNetwork

from src.audio.utils.constants import ASD_DIR

class ASDPipeline:
	def __init__(self, args):
		self.videoName = args.get("VIDEO_NAME","001")
		self.videoFolder = args.get("VIDEO_FOLDER","src/audio/ASD/demo")
		self.pretrainModel = args.get("PRETRAIN_ASD_MODEL","pretrain_TalkSet.model")
		self.pretrainModel = os.path.join(ASD_DIR, self.pretrainModel)
  
		self.nDataLoaderThread = args.get("N_DATA_LOADER_THREAD",32)
		self.facedetScale = args.get("FACE_DET_SCALE",0.25)
		self.minTrack = args.get("MIN_TRACK",10)
		self.numFailedDet = args.get("NUM_FAILED_DET",25)
		self.minFaceSize = args.get("MIN_FACE_SIZE",1)
		self.cropScale = args.get("CROP_SCALE",0.40)
		self.framesFaceTracking = args.get("FRAMES_FACE_TRACKING",1)
		self.start = args.get("START",0)
		self.duration = args.get("DURATION",0)
		self.thresholdSamePerson = args.get("THRESHOLD_SAME_PERSON",0.15)
		self.createTrackVideos = args.get("CREATE_TRACK_VIDEOS",True)
		self.includeVisualization = args.get("INCLUDE_VISUALIZATION",True)
  
  		# TODO: what exactly does this do?
		warnings.filterwarnings("ignore")
  
		# TODO: Do all the utils of beginning in init and then have them as class variables -> don't have the pass them all the time

		print("Only every xth frame will be analyzed for faster processing: ", self.framesFaceTracking)

		# Check whether GPU is available on cuda (NVIDIA), then mps (Macbook), otherwise use cpu
		self.device = get_device()

		# Download the pretrained model if not exist
		download_model(self.pretrainModel)
	
		self.videoPath, self.savePath = get_video_path(self.videoFolder, self.videoName)

		# Initialization
		self.pyaviPath = os.path.join(self.savePath, 'pyavi')
		self.pyworkPath = os.path.join(self.savePath, 'pywork')
	
		self.numFramesPerSec = get_frames_per_second(self.videoPath)
		self.totalFrames = get_num_total_frames(self.videoPath)
		
		# Extract the video from the start time to the duration
		self.videoPath = extract_video(self.pyaviPath, self.videoPath, self.duration, self.nDataLoaderThread, self.start, self.numFramesPerSec)	
     
		# TODO: Check if that's nevertheless necessary based on UX
		# if os.path.exists(savePath):
		# 	rmtree(savePath)

		if not os.path.exists(self.pyworkPath): # Save the results in this process by the pckl method
			os.makedirs(self.pyworkPath)
	
		if not os.path.exists(self.pyaviPath): # The path for the input video, input audio, output video
			os.makedirs(self.pyaviPath) 
   
		if os.path.exists(os.path.join(self.pyworkPath, 'faces.pckl')):
			with open(os.path.join(self.pyworkPath, 'faces.pckl'), 'rb') as f:
				self.faces = pickle.load(f)
		else:
			self.faces = None
   
		# Initialize the face detector
		self.face_detector = FaceDetector(self.device, self.videoPath, self.framesFaceTracking, self.facedetScale, self.pyworkPath)
  
		# Initialize the face tracker
		self.face_tracker = FaceTracker(self.numFailedDet, self.minTrack, self.minFaceSize)
  
		# Initialize the ASD network
		self.asd_network = ASDNetwork(self.device, self.pretrainModel, self.numFramesPerSec)

	# def extract_MFCC(self, file, outPath):
	# 	# CPU: extract mfcc
	# 	sr, audio = wavfile.read(file)
	# 	mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
	# 	featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
	# 	numpy.save(featuresPath, mfcc)



	def cutTrackVideos(self, trackSpeakingSegments, pyaviPath, videoPath, nDataLoaderThread) :
		# Using the trackSpeakingSegments, extract for each track the video segments from the original video (with moviepy)
		# Concatenate all the different subclip per track into one video
		# Go through each track
		for tidx, track in enumerate(trackSpeakingSegments):
			# Check whether the track is empty
			if len(track) == 0:
				continue

			# Only create the video if the output file does not exist
			cuttedFileName = os.path.join(pyaviPath, 'track_%s.mp4' % (tidx))
			if os.path.exists(cuttedFileName):
				continue
	
			# Create the list of subclips
			clips = []
			for segment in track:
				clips.append(VideoFileClip(videoPath).subclip(segment[0], segment[1]))
			# Concatenate the clips
			final_clip = concatenate_videoclips(clips)
			# Write the final video
			final_clip.write_videofile(cuttedFileName, threads=nDataLoaderThread, logger=None)

	def clusterTracks(self, tracks, faces, trackSpeakingSegments, trackSpeakingFaces, thresholdSamePerson):
		
		# Store the bounding boxes (x and y values) for each track by going through the face list
		trackListx = [[] for i in range(len(tracks))]
		trackListy = [[] for i in range(len(tracks))]
		for fidx, frame in enumerate(faces):
			for face in frame:
				trackListx[face['track']].append(face['x'])
				trackListy[face['track']].append(face['y'])
	
	# Calculate the average x and y value for each track
		trackAvgx = [[] for i in range(len(tracks))]
		for tidx, track in enumerate(trackListx):
			trackAvgx[tidx] = numpy.mean(track)
		trackAvgy = [[] for i in range(len(tracks))]
		for tidx, track in enumerate(trackListy):
			trackAvgy[tidx] = numpy.mean(track)    
		
		# Calculate the distance between the tracks
		trackDist = numpy.zeros((len(tracks), len(tracks)))
		for i in range(len(tracks)):
			for j in range(len(tracks)):
				trackDist[i,j] = math.sqrt((trackAvgx[i] - trackAvgx[j])**2 + (trackAvgy[i] - trackAvgy[j])**2)
	
		# Do make it independent of the image size in pixel, we need to normalize the distances
		trackDist = trackDist / numpy.max(trackDist)
	
	
		# Create a list of the tracks that are close to each other (if the distance is below defined threshold)
		trackClose = [[] for i in range(len(tracks))]
		for i in range(len(tracks)):
			for j in range(len(tracks)):
				if trackDist[i,j] < thresholdSamePerson and i != j:
					trackClose[i].append(j)
	
		# Check for the tracks that are close to each other if they are speaking at the same time (by checking if they if the lists in trackSpeakingFaces have shared elements/frames)
		# If no, store the track number in the list trackCloseSpeaking
		trackCloseSpeaking = []
		for i in range(len(tracks)):
			for j in range(len(trackClose[i])):
				if len(set(trackSpeakingFaces[i]) & set(trackSpeakingFaces[trackClose[i][j]])) == 0:
					trackCloseSpeaking.append(i)
					break
	
		# A cluster is a list of tracks that are close to each other and are not speaking at the same time
		# Create a list of clusters
		cluster = []
		for i in range(len(tracks)):
			if i in trackCloseSpeaking:
				cluster.append([i])
				for j in range(len(trackClose[i])):
					if trackClose[i][j] in trackCloseSpeaking:
						cluster[-1].append(trackClose[i][j])
	
	# Remove duplicates from the clusters (e.g. [1,2,3] and [3,2,1] are the same)
		unique_clusters = set()
		for i in cluster:
			i.sort()
			if tuple(i) in unique_clusters:
				cluster.remove(i)
			else:
				unique_clusters.add(tuple(i))
	
	# Print the unique clusters
		for i in unique_clusters:
			print("Tracks that belong together: ", i)
	
		return list(unique_clusters)

	def speakerSeparation(self, tracks, scores):
		
		all_faces = [[] for i in range(self.totalFrames)]
		
		# *Pick one track (e.g. one of the 7 as in in the sample)
		for tidx, track in enumerate(tracks):
			score = scores[tidx]
			# *Go through each frame in the selected track
			for fidx, frame in enumerate(track['track']['frame'].tolist()):
				s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
				s = numpy.mean(s)
				# *Store for each frame the bounding box and score (for each of the detected faces/tracks over time)
				all_faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	
		# From the faces list, remove the entries in each frame where the score is below 0
		SpeakingFaces = [[] for i in range(self.totalFrames)]
		for fidx, frame in enumerate(all_faces):
			SpeakingFaces[fidx] = [x for x in frame if x['score'] >= 0]
	
		# Create one list per track, where the entry is a list of the frames where the track is speaking
		trackSpeakingFaces = [[] for i in range(len(tracks))]
		for fidx, frame in enumerate(SpeakingFaces):
			for face in frame:
				trackSpeakingFaces[face['track']].append(fidx)
	
		# Create one list per track containing the start and end frame of the speaking segments
		trackSpeakingSegments = [[] for i in range(len(tracks))]
		for tidx, track in enumerate(trackSpeakingFaces):
			# *If the track is empty, skip it
			if len(track) == 0:
				continue
			
			trackSpeakingSegments[tidx] = [[track[0], track[0]]]
			for i in range(1, len(track)):
				if track[i] - track[i-1] == 1:
					trackSpeakingSegments[tidx][-1][1] = track[i]
				else:
					trackSpeakingSegments[tidx].append([track[i], track[i]])
		
		
		# Divide all number in trackSpeakingSegments by 25 (apart from 0) to get the time in seconds
		trackSpeakingSegments = [[[round(float(w/self.numFramesPerSec),2) if w != 0 else w for w in x] for x in y] for y in trackSpeakingSegments]

		# Check whether the start and end of a segment is the same (if yes, remove it) - throug rouding errors (conversion to seconds), this can happen
		for tidx, track in enumerate(trackSpeakingSegments):
			for i in range(len(track)):
				if track[i][0] == track[i][1]:
					trackSpeakingSegments[tidx].remove(track[i])

		if self.createTrackVideos:
			self.cutTrackVideos(trackSpeakingSegments, self.pyaviPath, self.videoPath, self.nDataLoaderThread)   

		# Sidenote: 
		# - x and y values are flipped (in contrast to normal convention)
		# - I make the assumption people do not change the place during one video I get as input 
		#   (-> if bounding boxes are close by and do not speak at the same time, then they are from the same person)
		# 	To correct for that, I would have to leverage face verification using the stored videos in pycrop 
	
		# Calculate tracks that belong together 
		sameTracks = self.clusterTracks(tracks, all_faces, trackSpeakingSegments, trackSpeakingFaces, self.thresholdSamePerson)
	
		# Calculate an rttm file based on trackSpeakingSegments (merge the speakers for the same tracks into one speaker)
		self.writerttm(self.pyaviPath, self.videoName, trackSpeakingSegments, sameTracks)
					

	def writerttm(self, pyaviPath, videoName, trackSpeakingSegments, sameTracks):
		# Create the rttm file
		with open(os.path.join(pyaviPath, videoName + '.rttm'), 'w') as rttmFile:
			for tidx, track in enumerate(trackSpeakingSegments):
				if len(track) == 0:
					continue
				# Go through each segment
				for sidx, segment in enumerate(track):
					# If the track belongs to cluster (in the sameTracks list), then write the lowest number (track) of that cluster to the rttm file
					check, lowestTrack = self.checkIfInClusterList(tidx, sameTracks)
					if check:
						rttmFile.write('SPEAKER %s 1 %s %s <NA> <NA> %s <NA> <NA>\n' % (videoName, segment[0],round(segment[1] - segment[0],2), lowestTrack))
					else:
						# Write the line to the rttm file, placeholder: file identifier, start time, duration, speaker ID
						rttmFile.write('SPEAKER %s 1 %s %s <NA> <NA> %s <NA> <NA>\n' % (videoName, segment[0], round(segment[1] - segment[0],2), tidx))
		
	def checkIfInClusterList(self, track, clusterList):
		for cluster in clusterList:
			if track in cluster:
				return True, min(cluster)
		return False, -1

	def visualization(self, tracks, scores):
		
		# TODO: Could be optimized here without using the storeFrames() function (directly use the video file)
		# storeFrames()
		
		# CPU: visulize the result for video format
		all_faces = [[] for i in range(self.totalFrames)]
		
		# *Pick one track (e.g. one of the 7 as in in the sample)
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
		cap = cv2.VideoCapture(self.videoPath)
		fw = int(cap.get(3))
		fh = int(cap.get(4))
		vOut = cv2.VideoWriter(os.path.join(self.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), self.numFramesPerSec, (fw,fh))
	
		# Instead of using the stored images in Pyframes, load the images from the video (which is stored at videoPath) and draw the bounding boxes there
		# CPU: visulize the result for video format
		# *Go through each frame
		for fidx in range(self.totalFrames):
			# *Load the frame from the video
			cap = cv2.VideoCapture(self.videoPath)
			cap.set(1, fidx)
			ret, image = cap.read()
			# *Within each frame go through each face and draw the bounding box
			for face in all_faces[fidx]:
				clr = colorDict[int((face['score'] >= 0))]
				txt = round(face['score'], 1)
				cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
				cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
			vOut.write(image)
		vOut.release()
	
		command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
			(os.path.join(self.pyaviPath, 'video_only.avi'), os.path.join(self.pyaviPath, 'audio.wav'), \
			self.nDataLoaderThread, os.path.join(self.pyaviPath,'video_out.avi'))) 
		output = subprocess.call(command, shell=True, stdout=None)


	# Pipeline for the ASD algorithm
	def run(self):
		# This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
		# ```
		# .
		# ├── pyavi
		# │   ├── audio.wav (Audio from input video)
		# │   ├── video.avi (Copy of the input video)
		# │   ├── video_only.avi (Output video without audio)
		# │   └── video_out.avi  (Output video with audio)
		# └── pywork
		#     ├── faces.pckl (face detection result)
		#     ├── scores.pckl (ASD result) - score values over time whether one speaks, for each detected face (video)
		#     └── tracks.pckl (face tracking result) - face bounding boxes over time for each detected face (video), per track x frames (e.g. in sample 7 tracks each ~500 frames)
		# ```
	
		# Assumption: If pickle files in pywork folder exist, ASD is done and all the other files exist (to re-run ASD delete pickle files)
		asd_done, scores, tracks = self._ASDPipeline__check_asd_done(self.pyworkPath)
		if asd_done:
			if self.includeVisualization == True:
				self.visualization(tracks, scores)
			self.speakerSeparation(tracks, scores)
			return
	
		# Extract audio from video
		audioFilePath = extract_audio_from_video(self.pyaviPath, self.videoPath, self.nDataLoaderThread)
	
		# Face detection (check for checkpoint first)
		if self.faces == None:
			self.faces = self.face_detector.s3fd_face_detection()
			sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(self.pyworkPath))
	
		# Face tracking
		# 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
		allTracks = []
		allTracks.extend(self.face_tracker.track_shot_face_tracker(self.faces))
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))

		# Crop all the tracks from the video (are stored in CPU memory)
		vidTracks, facesAllTracks = crop_tracks_from_videos_parallel(allTracks, self.videoPath, self.totalFrames, self.framesFaceTracking, self.cropScale, self.device)

		# TODO: Active Speaker Detection class (build wrapper to get easily also integrate other active speaker detection methods if necessary)
		# Active Speaker Detection by TalkNet
		scores = self.asd_network.talknet_network(allTracks, facesAllTracks, audioFilePath)
		savePath = os.path.join(self.pyworkPath, 'scores.pckl')
		with open(savePath, 'wb') as fil:
			pickle.dump(scores, fil)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %self.pyworkPath)
	
		savePath = os.path.join(self.pyworkPath, 'tracks.pckl')
		with open(savePath, 'wb') as fil:
			pickle.dump(vidTracks, fil)

		# Visualization, save the result as the new video	
		if self.includeVisualization == True:
			self.visualization(vidTracks, scores)
		self.speakerSeparation(vidTracks, scores)	

	def __check_asd_done(self, pyworkPath) -> tuple:
		# If pickle files exist in the pywork folder, then directly load the scores and tracks pickle files
		if os.path.exists(os.path.join(pyworkPath, 'scores.pckl')) and os.path.exists(os.path.join(pyworkPath, 'tracks.pckl')):
			with open(os.path.join(pyworkPath, 'scores.pckl'), 'rb') as f:
				scores = pickle.load(f)
			with open(os.path.join(pyworkPath, 'tracks.pckl'), 'rb') as f:
				tracks = pickle.load(f)
			return True, scores, tracks
		return False, None, None