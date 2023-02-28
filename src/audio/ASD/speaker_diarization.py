import numpy as np
import os
import math
# import matplotlib as plt

import cv2
#import insightface

from src.audio.ASD.cluster_tracks import ClusterTracks
from src.audio.ASD.utils.asd_pipeline_tools import cut_track_videos
from src.audio.ASD.utils.asd_pipeline_tools import write_to_terminal

class SpeakerDiarization:
    def __init__(self, pyavi_path, video_path, video_name, n_data_loader_thread, threshold_same_person, create_track_videos, 
                 total_frames, frames_per_second, save_path, faces_id_path, tracks_faces_clustering_path, crop_scale):
        self.pyavi_path = pyavi_path 
        self.video_path = video_path
        self.video_name = video_name
        self.n_data_loader_thread = n_data_loader_thread
        self.threshold_same_person = threshold_same_person
        self.create_track_videos = create_track_videos
        self.total_frames = total_frames
        self.frames_per_second = frames_per_second
        self.save_path = save_path
        self.faces_id_path = faces_id_path
        self.tracks_faces_clustering_path = tracks_faces_clustering_path
        self.crop_scale = crop_scale
        
        self.length_video = int(self.total_frames / self.frames_per_second)
        
        self.cluster_tracks = ClusterTracks(self.tracks_faces_clustering_path, self.video_path, self.crop_scale, self.threshold_same_person)
    
    def run(self, tracks, scores):
        
        write_to_terminal("Speaker diarization started")
        all_faces = [[] for i in range(self.total_frames)]
        
        # *Pick one track (e.g. one of the 7 as in in the sample)
        for tidx, track in enumerate(tracks):
            score = scores[tidx]
            # *Go through each frame in the selected track
            for fidx, frame in enumerate(track['track']['frame'].tolist()):
                s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
                s = np.mean(s)
                # *Store for each frame the bounding box and score (for each of the detected faces/tracks over time)
                all_faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})

        # From the faces list, remove the entries in each frame where the score is below 0
        speaking_faces = [[] for i in range(self.total_frames)]
        for fidx, frame in enumerate(all_faces):
            speaking_faces[fidx] = [x for x in frame if x['score'] >= 0]

        # Create one list per track, where the entry is a list of the frames where the track is speaking
        track_speaking_faces = [[] for i in range(len(tracks))]
        for fidx, frame in enumerate(speaking_faces):
            for face in frame:
                track_speaking_faces[face['track']].append(fidx)

        # Create one list per track containing the start and end frame of the speaking segments
        track_speaking_segments = [[] for i in range(len(tracks))]
        for tidx, track in enumerate(track_speaking_faces):
            # *If the track is empty, skip it
            if len(track) == 0:
                continue
            
            track_speaking_segments[tidx] = [[track[0], track[0]]]
            for i in range(1, len(track)):
                if track[i] - track[i-1] == 1:
                    track_speaking_segments[tidx][-1][1] = track[i]
                else:
                    track_speaking_segments[tidx].append([track[i], track[i]])
        
        
        # Divide all number in track_speaking_segments by 25 (apart from 0) to get the time in seconds
        track_speaking_segments = [[[round(float(w/self.frames_per_second),2) if w != 0 else w for w in x] for x in y] for y in track_speaking_segments]

        # Check whether the start and end of a segment is the same (if yes, remove it) - through rouding errors (conversion to seconds), this can happen
        for tidx, track in enumerate(track_speaking_segments):
            for i in range(len(track)):
                if track[i][0] == track[i][1]:
                    # track_speaking_segments[tidx].remove(track[i])
                    # Not removing them, rather setting them to 0
                    track_speaking_segments[tidx][i] = [0,0]
                    
        # Remove all segments where both start end end time are 0
        track_speaking_segments = [[x for x in y if x != [0,0]] for y in track_speaking_segments]


        if self.create_track_videos:
            cut_track_videos(track_speaking_segments, self.pyavi_path, self.video_path, self.n_data_loader_thread)   

        # Sidenote: 
        # - x and y values are flipped (in contrast to normal convention)
        # - I make the assumption people do not change the place during one video I get as input 
        #   (-> if bounding boxes are close by and do not speak at the same time, then they are from the same person)
        # 	To correct for that, I would have to leverage face verification
        
        # Calculate tracks that belong together based on face embeddings
        same_tracks = self.cluster_tracks.cluster_tracks_face_embedding(track_speaking_faces, tracks)
        
        # Store one image per track in a folder
        self.store_face_ids(self.faces_id_path, tracks, same_tracks)

        # Old clustering based on assumption, that people do not change the place during one video
        # same_tracks = self.cluster_tracks.cluster_tracks(tracks, all_faces, track_speaking_faces)

        # Calculate an rttm file based on trackSpeakingSegments (merge the speakers for the same tracks into one speaker)
        self.write_rttm(track_speaking_segments, same_tracks)   
        
        return
    
    def store_face_ids(self, faces_id_path, tracks, same_tracks) -> None:
        # Store in a dict for each track the track id, the frame number of the first frame and the corresponding bounding box
        track_bboxes = [None for i in range(len(tracks))]
        for tidx, track in enumerate(tracks):
            # Take the one frame of every track and store it
            frame_number = -10
            track_bboxes[tidx] = [track['track']['frame'][frame_number] ,track['proc_track']['s'][frame_number], track['proc_track']['x'][frame_number], track['proc_track']['y'][frame_number]]
            
        # From the list same_tracks, extract from every list it contains the first element and store it in a list
        cluster_ids = [x[0] for x in same_tracks]    
            
        # Using faces_id dict, now crop for each track the bounding box at the corresponding frame and save it as an image            
        for tidx, track_bbox in enumerate(track_bboxes):
            
            # Only store one picture per recognized cluster
            if tidx not in cluster_ids:
                continue
            
            cap = cv2.VideoCapture(self.video_path)
            cap.set(1, track_bbox[0])
            ret, frame = cap.read()
            cap.release()
            
            # Using the bbox from the faces_id dict to crop the image
            
            s = track_bbox[1]
            x = track_bbox[2]
            y = track_bbox[3]
            
            cs  = self.crop_scale
            bs  = s   # Detection box size
            bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
            frame = np.pad(frame, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
            my  = y + bsi  # BBox center Y
            mx  = x + bsi  # BBox center X
            face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]

            
            # Save the image            
            cv2.imwrite(os.path.join(faces_id_path, str(tidx) + ".jpg"), cv2.resize(face, (224, 224)))
        
        
    def write_rttm(self, track_speaking_segments, same_tracks):
        # Create the rttm file
        
        file_name = self.video_name + "_" + str(self.length_video) + "s.rttm"
        
        with open(os.path.join(self.save_path, file_name), 'w') as rttmFile:
            for tidx, track in enumerate(track_speaking_segments):
                if len(track) == 0:
                    continue
                # Go through each segment
                check, lowestTrack = self.check_if_in_cluster_list(tidx, same_tracks)
                for segment in track:
                    # If the track belongs to cluster (in the sameTracks list), then write the lowest number (track) of that cluster to the rttm file
                    if check:
                        rttmFile.write('SPEAKER %s 1 %s %s <NA> <NA> %s <NA> <NA>\n' % (self.video_name, segment[0],round(segment[1] - segment[0],2), lowestTrack))
                    else:
                        # Write the line to the rttm file, placeholder: file identifier, start time, duration, speaker ID
                        rttmFile.write('SPEAKER %s 1 %s %s <NA> <NA> %s <NA> <NA>\n' % (self.video_name, segment[0], round(segment[1] - segment[0],2), tidx))
    
    
    def check_if_in_cluster_list(self, track, clusterList):
        for cluster in clusterList:
            if track in cluster:
                return True, min(cluster)
        return False, -1