import numpy
import os
import math
import cv2
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.audio.ASD.utils.asd_pipeline_tools import cut_track_videos
from src.audio.ASD.utils.asd_pipeline_tools import write_to_terminal

from src.audio.utils.constants import VIDEOS_DIR

class SpeakerDiarization:
    def __init__(self, pyavi_path, video_path, video_name, n_data_loader_thread, threshold_same_person, create_track_videos, total_frames, frames_per_second, save_path, faces_id_path, crop_scale):
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
        self.crop_scale = crop_scale
        
        self.length_video = int(self.total_frames / self.frames_per_second)
    
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

        # Calculate tracks that belong together 
        same_tracks = self.cluster_tracks(tracks, all_faces, track_speaking_faces)

        # Calculate an rttm file based on trackSpeakingSegments (merge the speakers for the same tracks into one speaker)
        self.write_rttm(track_speaking_segments, same_tracks)   
        
        self.store_face_ids(self.faces_id_path, tracks)
        
        return self.length_video
    
    # TODO: store one face per ID (track), or only the clustered ones? (it might be confusing for user to see more ids then recognized, but it is helpful more me to debug)
    def store_face_ids(self, faces_id_path, tracks) -> None:
        # Store in a dict for each track the track id, the frame number of the first frame and the corresponding bounding box
        track_bboxes = [None for i in range(len(tracks))]
        for tidx, track in enumerate(tracks):
            # Take the first frame of every track and store it
            frame_number = 0
            track_bboxes[tidx] = [track['track']['frame'][frame_number] ,track['proc_track']['s'][frame_number], track['proc_track']['x'][frame_number], track['proc_track']['y'][frame_number]]
            
        # Using faces_id dict, now crop for each track the bounding box at the corresponding frame and save it as an image            
        for tidx, track_bbox in enumerate(track_bboxes):
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
            frame = numpy.pad(frame, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
            my  = y + bsi  # BBox center Y
            mx  = x + bsi  # BBox center X
            face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]

            
            # Save the image
            cv2.imwrite(os.path.join(faces_id_path, str(tidx) + ".jpg"), cv2.resize(face, (224, 224)))
        
        
    def write_rttm(self, track_speaking_segments, same_tracks):
        # Create the rttm file
        
        file_name = self.video_name + "_" + str(self.length_video) + ".rttm"
        
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

                
    def cluster_tracks(self, tracks, faces, track_speaking_faces):
        
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

        
        dbscan = DBSCAN(eps= self.threshold_same_person, min_samples=2)
        
        # Create the datapoints for the DBSCAN algorithm
        x = []
        for i in range(len(tracks)):
            x.append([track_avg_x[i], track_avg_y[i]])
            
        # create scaler object and fit to data
        scaler = StandardScaler()
        scaler.fit(x)

        # transform data using scaler
        x_transformed = scaler.transform(x)
        
        # Calculate the pairwise distances between the scaled data points
        track_dist = numpy.zeros((len(tracks), len(tracks)))
        for i in range(len(tracks)):
            for j in range(len(tracks)):
                track_dist[i,j] = math.sqrt((x_transformed[i,0] - x_transformed[j,0])**2 + (x_transformed[i,1] - x_transformed[j,1])**2)
        
        # # Using matplotlib, plot the data points (label each point with the index of the track)
        # for i in range(len(x_transformed)):
        #     plt.scatter(x_transformed[i,0], x_transformed[i,1], label=i)  
        # plt.legend()  
        # plt.show()

        # perform clustering
        labels = dbscan.fit_predict(x_transformed)
        
        # Print the indices of each cluster (if the label is -1, then print it as a cluster with only one index)
        unique_labels = numpy.unique(labels)
        for label in unique_labels:
            if label == -1:
                print("Outlier:")
                outlier_indices = numpy.where(labels == label)[0]
                for i in outlier_indices:
                    if track_speaking_faces[i] != []:
                        print(f"\t\t{i}")
            else:
                print(f"Cluster {label}:")
                cluster_indices = numpy.where(labels == label)[0]
                for i in cluster_indices:
                    if track_speaking_faces[i] != []:
                        print(f"\t\t{i}")
        
            

        # Store the indices of the tracks that belong to the same cluster
        clusters = []
        for i in range(0, labels.max() + 1):
            clusters.append(numpy.where(labels == i)[0])

                
        # Create a copy of the clusters list where only the indices of the tracks that are speaking are stored (the have at least one element in the track_speaking_faces list)
        speaking_clusters = []
        for i in range(len(clusters)):
            speaking_clusters.append([])
            for j in range(len(clusters[i])):
                if len(track_speaking_faces[clusters[i][j]]) > 0:
                    speaking_clusters[i].append(clusters[i][j])
                    
        # If one cluster has only one track, then it is not a cluster, so delete it
        speaking_clusters = [x for x in speaking_clusters if len(x) > 1]

        return speaking_clusters
    
    
    def check_if_in_cluster_list(self, track, clusterList):
        for cluster in clusterList:
            if track in cluster:
                return True, min(cluster)
        return False, -1