import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features, moviepy
import multiprocessing
from pydub import AudioSegment
import torchvision.transforms as transforms

# TODO: Add pydub to requirements.txt

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

# Disabled scene detection for now, because cutted teamwork videos have no change in scene 
# if you want to add it later, have a look at the original repo: https://github.com/TaoRuijie/TalkNet-ASD

from model.faceDetector.s3fd import S3FD
from talkNet import talkNet

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "TalkNet Demo or Columnbia ASD Evaluation")

parser.add_argument('--videoName',             type=str, default="001",   help='Demo video name')
parser.add_argument('--videoFolder',           type=str, default="demo",  help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel',         type=str, default="pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')

parser.add_argument('--nDataLoaderThread',     type=int,   default=32,   help='Number of workers')
parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')

parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each scene and track')
parser.add_argument('--numFailedDet',          type=int,   default=25,   help='Number of missed detections allowed before tracking is stopped (e.g. 25 fps -> 1 sec)')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')

parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')
parser.add_argument('--framesFaceTracking',    type=float, default=10, help='To speed up the face tracking, we only track the face in every x frames')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

parser.add_argument('--thresholdSamePerson',   type=str, default=0.15,  help='If two face tracks (see folder pycrop) are close together (-> below that threshold) and are not speaking at the same time, then it is the same person')
parser.add_argument('--createTrackVideos',     type=bool, default=True,  help='If enabled, it will create a video for each track, where only the segments where the person is speaking are included')

parser.add_argument('--includeVisualization', type=bool, default=True,  help='If enabled, it will create a video where you can see the speaking person highlighted (e.g. used for debugging)')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Detected device (GPU/CPU): ", device)

if os.path.isfile(args.pretrainModel) == False: # Download the pretrained model
    Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
    cmd = "gdown --id %s -O %s"%(Link, args.pretrainModel)
    subprocess.call(cmd, shell=True, stdout=None)

def get_video_path(args):
	args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
	args.savePath = os.path.join(args.videoFolder, args.videoName)


def inference_video(args):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	DET = S3FD(device=device)
  
	# Instead of using the stored images in Pyframes, load the images from the video (which is stored at args.videoFilePath) and go with the detection through each frame
	cap = cv2.VideoCapture(args.videoFilePath)
	
	# TODO: Interpolate linearly between the bounding boxes of the previous and next frame
	# Instead of going through every frame for the face detection, we go through every xth (e.g. 10th) frame and then use the bounding boxes from the previous frame to track the faces in the next frames
	# This is done to reduce the number of frames that need to be processed for the face detection
	dets = []
	fidx = 0
	while(cap.isOpened()):
		ret, image = cap.read()
		if ret == False:
			break
		if fidx%args.framesFaceTracking == 0:
			imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
			dets.append([])
			for bbox in bboxes:
				dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})
		else:
			dets.append([])
			for bbox in dets[-2]:
				dets[-1].append({'frame':fidx, 'bbox':bbox['bbox'], 'conf':bbox['conf']})
		sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
		fidx += 1
	
  
	savePath = os.path.join(args.pyworkPath,'faces.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(dets, fil)
	return dets

def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def track_shot(args, sceneFaces):
	# CPU: Face tracking
	iouThres  = 0.5     # Minimum IOU between consecutive face detections
	tracks    = []
	while True:
		track     = []
		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
						continue
				else:
					break
		if track == []:
			break
		elif len(track) > args.minTrack:
			frameNum    = numpy.array([ f['frame'] for f in track ])
			bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
			frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
			bboxesI    = []
			for ij in range(0,4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))
			bboxesI  = numpy.stack(bboxesI, axis=1)
			if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
				tracks.append({'frame':frameI,'bbox':bboxesI})
	return tracks


def crop_track_faster(args, track):

	dets = {'x':[], 'y':[], 's':[]}
	# Instead of going through every track['bbox'] for the calculation of the dets variable, we go through every 10th value and then use the dets values from the previous one to for the next 9 values 
	for fidx, det in enumerate(track['bbox']):
		if fidx%args.framesFaceTracking == 0:
			dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
			dets['y'].append((det[1]+det[3])/2) # crop center x
			dets['x'].append((det[0]+det[2])/2) # crop center y
		else:
			dets['s'].append(dets['s'][-1])
			dets['y'].append(dets['y'][-1])
			dets['x'].append(dets['x'][-1])

	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

	vIn = cv2.VideoCapture(args.videoFilePath)
	num_frames = len(track['frame'])

	# Create an empty array for the faces
	faces = torch.zeros((num_frames, 112, 112), dtype=torch.float32)
 
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
 
	for fidx, frame in enumerate(track['frame']):
		cs  = args.cropScale
		bs  = dets['s'][fidx]   # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount     
		
		vIn.set(cv2.CAP_PROP_POS_FRAMES, frame)
		ret, image = vIn.read()
		
		# Pad the image with constant values
		image = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		
		# Crop the face from the image
		face = image[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		
		# Apply the transformations
		face = transform(face)
		# Have to get permuted as numpy uses (H, W, C) and pytorch uses (C, H, W)
		
		faces[fidx, :, :] = face[0, :, :]
		face = face.to(device)

	vIn.release()

	return {'track':track, 'proc_track':dets}, faces


def crop_track_fastest(args, track):

	dets = {'x':[], 'y':[], 's':[]}
	# Instead of going through every track['bbox'] for the calculation of the dets variable, we go through every 10th value and then use the dets values from the previous one to for the next 9 values 
	for fidx, det in enumerate(track['bbox']):
		if fidx%args.framesFaceTracking == 0:
			dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
			dets['y'].append((det[1]+det[3])/2) # crop center x
			dets['x'].append((det[0]+det[2])/2) # crop center y
		else:
			dets['s'].append(dets['s'][-1])
			dets['y'].append(dets['y'][-1])
			dets['x'].append(dets['x'][-1])

	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
 
	vIn = cv2.VideoCapture(args.videoFilePath)
	num_frames = len(track['frame'])

	# Create an empty array for the faces
	faces = torch.zeros((num_frames, 112, 112), dtype=torch.float32)

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

	# Number of frames to process at once
	batch_size = 32

	# Loop over the frames in batches
	for start in range(0, num_frames, batch_size):
		end = min(start + batch_size, num_frames)
		batch_indices = range(start, end)

		batch_frames = track['frame'][batch_indices]
		batch_dets_y = dets['y'][batch_indices]
		batch_dets_x = dets['x'][batch_indices]
		batch_dets_s = dets['s'][batch_indices]

		batch_images = []
		for frame, my, mx, bs in zip(batch_frames, batch_dets_y, batch_dets_x, batch_dets_s):
			vIn.set(cv2.CAP_PROP_POS_FRAMES, frame)
			ret, image = vIn.read()

			# Pad the image with constant values
			bsi = int(bs * (1 + 2 * args.cropScale))
			image = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))

			# Crop the face from the image
			my += bsi  # BBox center Y
			mx += bsi  # BBox center X
			face = image[int(my-bs):int(my+bs*(1+2*args.cropScale)),int(mx-bs*(1+args.cropScale)):int(mx+bs*(1+args.cropScale))]

			batch_images.append(face)

		batch_images = torch.stack([transform(image) for image in batch_images], dim=0)
		batch_images = batch_images[:,0,:,:].to(device)

		faces[start:end, :, :] = batch_images.to(device)

	vIn.release()

	return {'track':track, 'proc_track':dets}, faces

def extract_MFCC(file, outPath):
	# CPU: extract mfcc
	sr, audio = wavfile.read(file)
	mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
	featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
	numpy.save(featuresPath, mfcc)
 
def extract_audio(audio_file, track, args):
	audioStart  = (track['frame'][0]) / args.numFramesPerSec
	audioEnd    = (track['frame'][-1]+1) / args.numFramesPerSec
	sound = AudioSegment.from_wav(audio_file)
    
	segment = sound[audioStart*1000:audioEnd*1000]
	samplerate = segment.frame_rate
	trans_segment = numpy.array(segment.get_array_of_samples(), dtype=numpy.int16)
	return trans_segment, samplerate

def evaluate_network(allTracks, args):
	# GPU: active speaker detection by pretrained TalkNet
	s = talkNet().to(device)
	s.loadParameters(args.pretrainModel)
	sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
	s.eval()	

	allScores, vidTracks = [], []
	# durationSet = {1,2,4,6} # To make the result more reliable
	durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
	for tidx, track in tqdm.tqdm(enumerate(allTracks), total = len(allTracks)):
  
		segment, samplerate = extract_audio(args.audioFilePath, track, args)
		audioFeature = python_speech_features.mfcc(segment, samplerate, numcep = 13, winlen = 0.025, winstep = 0.010)
  
		# Instead of saving the cropped the video, call the crop_track function to return the faces (without saving them)
		# * Problem: The model might have been trained with compressed image data (as I directly load them and don't save them as intermediate step, my images are slightly different)
		# * Rounding is different (so sometimes numbers are slightly different) + changes in Color as OpenCV conversion works differently (but videos look the same)
		# old_trackDict, old_videoFeature = crop_track_faster(args, track)
		trackDict, videoFeature = crop_track_fastest(args, track)
		vidTracks.append(trackDict)
  
		length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
		audioFeature = audioFeature[:int(round(length * 100)),:]
		videoFeature = videoFeature[:int(round(length * args.numFramesPerSec)),:,:]
		allScore = [] # Evaluation use TalkNet
		for duration in durationSet:
			batchSize = int(math.ceil(length / duration))
			scores = []
			with torch.no_grad():
				for i in range(batchSize):
					inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).to(device)
					inputV = videoFeature[i * duration * args.numFramesPerSec: (i+1) * duration * args.numFramesPerSec,:,:].unsqueeze(0).to(device)
					embedA = s.model.forward_audio_frontend(inputA).to(device)
					embedV = s.model.forward_visual_frontend(inputV).to(device)
					embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
					out = s.model.forward_audio_visual_backend(embedA, embedV)
					score = s.lossAV.forward(out, labels = None)
					scores.extend(score)
			allScore.append(scores)
		allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
		allScores.append(allScore)	
	return allScores, vidTracks

def cutTrackVideos(trackSpeakingSegments, args):
    # Using the trackSpeakingSegments, extract for each track the video segments from the original video (with moviepy)
	# Concatenate all the different subclip per track into one video
	# Go through each track
	for tidx, track in enumerate(trackSpeakingSegments):
		# Check whether the track is empty
		if len(track) == 0:
			continue

		# Only create the video if the output file does not exist
		cuttedFileName = os.path.join(args.pyaviPath, 'track_%s.mp4' % (tidx))
		if os.path.exists(cuttedFileName):
			continue
   
		# Create the list of subclips
		clips = []
		for segment in track:
			clips.append(VideoFileClip(args.videoPath).subclip(segment[0], segment[1]))
		# Concatenate the clips
		final_clip = concatenate_videoclips(clips)
		# Write the final video
		final_clip.write_videofile(cuttedFileName, threads=args.nDataLoaderThread, logger=None)

def clusterTracks(tracks, faces, trackSpeakingSegments, trackSpeakingFaces, args):
    
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
			if trackDist[i,j] < args.thresholdSamePerson and i != j:
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

def speakerSeparation(tracks, scores, args):
    # CPU: visulize the result for video format
    
	faces = [[] for i in range(args.totalFrames)]
	
	# *Pick one track (e.g. one of the 7 as in in the sample)
	for tidx, track in enumerate(tracks):
		score = scores[tidx]
		# *Go through each frame in the selected track
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
			s = numpy.mean(s)
			# *Store for each frame the bounding box and score (for each of the detected faces/tracks over time)
			faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
 
 	# From the faces list, remove the entries in each frame where the score is below 0
	SpeakingFaces = [[] for i in range(args.totalFrames)]
	for fidx, frame in enumerate(faces):
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
	trackSpeakingSegments = [[[round(float(w/args.numFramesPerSec),2) if w != 0 else w for w in x] for x in y] for y in trackSpeakingSegments]

	if args.createTrackVideos:
		cutTrackVideos(trackSpeakingSegments, args)   

	# Sidenote: 
 	# - x and y values are flipped (in contrast to normal convention)
	# - I make the assumption people do not change the place during one video I get as input 
    #   (-> if bounding boxes are close by and do not speak at the same time, then they are from the same person)
 	# 	To correct for that, I would have to leverage face verification using the stored videos in pycrop 
  
	# Calculate tracks that belong together 
	sameTracks = clusterTracks(tracks, faces, trackSpeakingSegments, trackSpeakingFaces, args)
 
	# Calculate an rttm file based on trackSpeakingSegments (merge the speakers for the same tracks into one speaker)
	writerttm(args, trackSpeakingSegments, sameTracks)
				

def writerttm(args, trackSpeakingSegments, sameTracks):
	# Create the rttm file
	with open(os.path.join(args.pyaviPath, args.videoName + '.rttm'), 'w') as rttmFile:
		for tidx, track in enumerate(trackSpeakingSegments):
			if len(track) == 0:
				continue
			# Go through each segment
			for sidx, segment in enumerate(track):
				# If the track belongs to cluster (in the sameTracks list), then write the lowest number (track) of that cluster to the rttm file
				check, lowestTrack = checkIfInClusterList(tidx, sameTracks)
				if check:
					rttmFile.write('SPEAKER %s 1 %s %s <NA> <NA> %s <NA> <NA>\n' % (args.videoName, segment[0],round(segment[1] - segment[0],2), lowestTrack))
				else:
					# Write the line to the rttm file, placeholder: file identifier, start time, duration, speaker ID
					rttmFile.write('SPEAKER %s 1 %s %s <NA> <NA> %s <NA> <NA>\n' % (args.videoName, segment[0], round(segment[1] - segment[0],2), tidx))
	
def checkIfInClusterList(track, clusterList):
	for cluster in clusterList:
		if track in cluster:
			return True, min(cluster)
	return False, -1

def visualization(tracks, scores, args):
    
    # TODO: Could be optimized here without using the storeFrames() function (directly use the video file)
    # storeFrames()
    
	# CPU: visulize the result for video format
	faces = [[] for i in range(args.totalFrames)]
	
	# *Pick one track (e.g. one of the 7 as in in the sample)
	for tidx, track in enumerate(tracks):
		score = scores[tidx]
		# *Go through each frame in the selected track
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
			s = numpy.mean(s)
			# *Store for each frame the bounding box and score (for each of the detected faces/tracks over time)
			faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
 
	colorDict = {0: 0, 1: 255}
	# Get height and width in pixel of the video
	cap = cv2.VideoCapture(args.videoFilePath)
	fw = int(cap.get(3))
	fh = int(cap.get(4))
	vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), args.numFramesPerSec, (fw,fh))
 
 	# Instead of using the stored images in Pyframes, load the images from the video (which is stored at args.videoFilePath) and draw the bounding boxes there
	# CPU: visulize the result for video format
	# *Go through each frame
	for fidx in range(args.totalFrames):
		# *Load the frame from the video
		cap = cv2.VideoCapture(args.videoFilePath)
		cap.set(1, fidx)
		ret, image = cap.read()
		# *Within each frame go through each face and draw the bounding box
		for face in faces[fidx]:
			clr = colorDict[int((face['score'] >= 0))]
			txt = round(face['score'], 1)
			cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
			cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
		vOut.write(image)
	vOut.release()
 
	command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
		(os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'), \
		args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.avi'))) 
	output = subprocess.call(command, shell=True, stdout=None)
 
# TODO: To be updated, if use again 
# def crop_tracks_in_parallel(args, allTracks):
#     with multiprocessing.Pool() as pool:
#         vidTracks = [pool.apply_async(crop_track, args=(args, track))
#                      for ii, track in enumerate(allTracks)]
#         vidTracks = [result.get() for result in tqdm.tqdm(vidTracks, total=len(allTracks))]
#     return vidTracks

 
def get_fps(video_path):
	video = cv2.VideoCapture(video_path)
	fps = video.get(cv2.CAP_PROP_FPS)
	video.release()
	return fps

# Main function
def main():
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
 
	get_video_path(args)

	# Initialization 
	args.pyaviPath = os.path.join(args.savePath, 'pyavi')
	args.pyworkPath = os.path.join(args.savePath, 'pywork')
 
	args.numFramesPerSec = int(get_fps(args.videoPath))
	print("Frames per second for the detected video: ", args.numFramesPerSec)
	
 	# Cut the video if necessary
	args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
	# If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
	if args.duration == 0:
		args.videoFilePath = args.videoPath
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Video will not be cutted and remains in %s \r\n" %(args.videoFilePath))
	else:
		command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r %d %s -loglevel panic" % \
			(args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.numFramesPerSec, args.videoFilePath))
		subprocess.call(command, shell=True, stdout=None)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" %(args.videoFilePath))
	
    # Get the number of total frames of the video stored in videoFilePath
	args.totalFrames = int(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'default=nokey=1:noprint_wrappers=1', args.videoFilePath]))
	print("Total frames for the detected video: ", args.totalFrames)
 
	# Assumption: If pickle files in pywork folder exist, preprocessing is done and all the other files exist (to rerun delete pickle files)
	# If 4 pickle files exist in the pywork folder, then directly load the scores and tracks pickle files
	if os.path.exists(os.path.join(args.pyworkPath, 'scores.pckl')) and os.path.exists(os.path.join(args.pyworkPath, 'tracks.pckl')):
		with open(os.path.join(args.pyworkPath, 'scores.pckl'), 'rb') as f:
			scores = pickle.load(f)
		with open(os.path.join(args.pyworkPath, 'tracks.pckl'), 'rb') as f:
			tracks = pickle.load(f)
		if args.includeVisualization == True:
			visualization(tracks, scores, args)
		speakerSeparation(tracks, scores, args)
		return

 
	if os.path.exists(args.savePath):
		rmtree(args.savePath)
	os.makedirs(args.pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
	os.makedirs(args.pyworkPath, exist_ok = True) # Save the results in this process by the pckl method
 
	# Extract audio
	# TODO: Can't I just get the audio from the original video and storing it in a variable instead of saving it in a file?
	args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
	command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
		(args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
	subprocess.call(command, shell=True, stdout=None)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(args.audioFilePath))

	# Face detection for the video frames
	faces = inference_video(args)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(args.pyworkPath))
 
	allTracks = []
	allTracks.extend(track_shot(args, faces)) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))

	# Active Speaker Detection by TalkNet
	scores, vidTracks = evaluate_network(allTracks, args)
	savePath = os.path.join(args.pyworkPath, 'scores.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(scores, fil)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %args.pyworkPath)
 
	savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(vidTracks, fil)


	# Visualization, save the result as the new video	
	if args.includeVisualization == True:
		visualization(vidTracks, scores, args)
	speakerSeparation(vidTracks, scores, args)	

if __name__ == '__main__':
    main()
