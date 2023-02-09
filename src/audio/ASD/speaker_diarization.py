import numpy
import os
import math

from src.audio.ASD.utils.asd_pipeline_tools import cut_track_videos
from src.audio.ASD.utils.asd_pipeline_tools import write_to_terminal

class SpeakerDiarization:
    def __init__(self, pyavi_path, video_path, video_name, n_data_loader_thread, threshold_same_person, create_track_videos, total_frames, frames_per_second):
        self.pyavi_path = pyavi_path 
        self.video_path = video_path
        self.video_name = video_name
        self.n_data_loader_thread = n_data_loader_thread
        self.threshold_same_person = threshold_same_person
        self.create_track_videos = create_track_videos
        self.total_frames = total_frames
        self.frames_per_second = frames_per_second
    
    def run(self, tracks, scores):
        
        write_to_terminal("Speaker diarization started")
        all_faces = [[] for i in range(self.total_frames)]
        
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

        # Check whether the start and end of a segment is the same (if yes, remove it) - throug rouding errors (conversion to seconds), this can happen
        for tidx, track in enumerate(track_speaking_segments):
            for i in range(len(track)):
                if track[i][0] == track[i][1]:
                    track_speaking_segments[tidx].remove(track[i])

        if self.create_track_videos:
            cut_track_videos(track_speaking_segments, self.pyavi_path, self.video_path, self.n_data_loader_thread)   

        # Sidenote: 
        # - x and y values are flipped (in contrast to normal convention)
        # - I make the assumption people do not change the place during one video I get as input 
        #   (-> if bounding boxes are close by and do not speak at the same time, then they are from the same person)
        # 	To correct for that, I would have to leverage face verification using the stored videos in pycrop 

        # Calculate tracks that belong together 
        same_tracks = self.cluster_tracks(tracks, all_faces, track_speaking_faces, self.threshold_same_person)

        # Calculate an rttm file based on trackSpeakingSegments (merge the speakers for the same tracks into one speaker)
        self.write_rttm(self.pyavi_path, self.video_name, track_speaking_segments, same_tracks)   
        
        
    def write_rttm(self, pyavi_path, video_name, track_speaking_segments, same_tracks):
        # Create the rttm file
        with open(os.path.join(pyavi_path, video_name + '.rttm'), 'w') as rttmFile:
            for tidx, track in enumerate(track_speaking_segments):
                if len(track) == 0:
                    continue
                # Go through each segment
                for sidx, segment in enumerate(track):
                    # If the track belongs to cluster (in the sameTracks list), then write the lowest number (track) of that cluster to the rttm file
                    check, lowestTrack = self.check_if_in_cluster_list(tidx, same_tracks)
                    if check:
                        rttmFile.write('SPEAKER %s 1 %s %s <NA> <NA> %s <NA> <NA>\n' % (video_name, segment[0],round(segment[1] - segment[0],2), lowestTrack))
                    else:
                        # Write the line to the rttm file, placeholder: file identifier, start time, duration, speaker ID
                        rttmFile.write('SPEAKER %s 1 %s %s <NA> <NA> %s <NA> <NA>\n' % (video_name, segment[0], round(segment[1] - segment[0],2), tidx))

                
    def cluster_tracks(self, tracks, faces, track_speaking_faces, threshold_same_person):
        
        # Store the bounding boxes (x and y values) for each track by going through the face list
        track_list_x = [[] for i in range(len(tracks))]
        track_list_y = [[] for i in range(len(tracks))]
        for fidx, frame in enumerate(faces):
            for face in frame:
                track_list_x[face['track']].append(face['x'])
                track_list_y[face['track']].append(face['y'])

        # Calculate the average x and y value for each track
        track_avg_x = [[] for i in range(len(tracks))]
        for tidx, track in enumerate(track_list_x):
            track_avg_x[tidx] = numpy.mean(track)
        track_avg_y = [[] for i in range(len(tracks))]
        for tidx, track in enumerate(track_list_y):
            track_avg_y[tidx] = numpy.mean(track)    

        # Calculate the distance between the tracks
        track_dist = numpy.zeros((len(tracks), len(tracks)))
        for i in range(len(tracks)):
            for j in range(len(tracks)):
                track_dist[i,j] = math.sqrt((track_avg_x[i] - track_avg_x[j])**2 + (track_avg_y[i] - track_avg_y[j])**2)

        # Do make it independent of the image size in pixel, we need to normalize the distances
        track_dist = track_dist / numpy.max(track_dist)


        # Create a list of the tracks that are close to each other (if the distance is below defined threshold)
        track_close = [[] for i in range(len(tracks))]
        for i in range(len(tracks)):
            for j in range(len(tracks)):
                if track_dist[i,j] < threshold_same_person and i != j:
                    track_close[i].append(j)

        # Check for the tracks that are close to each other if they are speaking at the same time (by checking if they if the lists in trackSpeakingFaces have shared elements/frames)
        # If no, store the track number in the list trackCloseSpeaking
        track_close_speaking = []
        for i in range(len(tracks)):
            for j in range(len(track_close[i])):
                if len(set(track_speaking_faces[i]) & set(track_speaking_faces[track_close[i][j]])) == 0:
                    track_close_speaking.append(i)
                    break

        # A cluster is a list of tracks that are close to each other and are not speaking at the same time
        # Create a list of clusters
        cluster = []
        for i in range(len(tracks)):
            if i in track_close_speaking:
                cluster.append([i])
                for j in range(len(track_close[i])):
                    if track_close[i][j] in track_close_speaking:
                        cluster[-1].append(track_close[i][j])

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
            write_to_terminal("Tracks that belong together: " + str(i))

        return list(unique_clusters)
    
    
    def check_if_in_cluster_list(self, track, clusterList):
        for cluster in clusterList:
            if track in cluster:
                return True, min(cluster)
        return False, -1