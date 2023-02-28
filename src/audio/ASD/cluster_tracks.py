import os
import cv2
import numpy as np
import math
import pickle
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from insightface.app import FaceAnalysis

class ClusterTracks:
    def __init__(self, tracks_faces_clustering_path, video_path, crop_scale) -> None:
        self.tracks_faces_clustering_path = tracks_faces_clustering_path
        self.video_path = video_path
        self.crop_scale = crop_scale
    
    def store_face_verification_track_images(self, tracks) -> None:
        # Store in a dict for each track the track id, the frame number of the first frame and the corresponding bounding box
        # track_bboxes = [None for i in range(len(tracks))]
        # for tidx, track in enumerate(tracks):
        #     # Take the one frame of every track and store it
        #     frame_number = -10
        #     track_bboxes[tidx] = [track['track']['frame'][frame_number] ,track['proc_track']['s'][frame_number], track['proc_track']['x'][frame_number], track['proc_track']['y'][frame_number]]
         
        # Instead of getting just one frame per track, get 5 random frames per track
        track_bboxes = []
        for tidx, track in enumerate(tracks):
            # Take the one frame of every track and store it
            
            # Get 5 random numbers from the array track['track']['frame']
            frame_numbers = np.random.choice(track['track']['frame'], 5, replace=False)
            
            # Get the indices of the frame numbers in the track['track']['frame'] array
            frame_numbers = np.searchsorted(track['track']['frame'], frame_numbers)
            
            track_bboxes.append([])
            
            for frame_number in frame_numbers:
                track_bboxes[tidx].append([track['track']['frame'][frame_number] ,track['proc_track']['s'][frame_number], track['proc_track']['x'][frame_number], track['proc_track']['y'][frame_number]])
         
            
        # Using faces_id dict, now crop for each track the bounding box at the corresponding frame and save it as an image            
        for tidx, track_bbox in enumerate(track_bboxes):
            
            # Make a 4 digit number of out the ID (e.g. 1 -> 001)
            track_id = str(tidx)
            if len(track_id) == 1:
                track_id = "000" + track_id
            elif len(track_id) == 2:
                track_id = "00" + track_id
            elif len(track_id) == 3:    
                track_id = "0" + track_id
            
            # Create for each track a folder  
            track_folder_path = os.path.join(self.tracks_faces_clustering_path, track_id)
            if not os.path.exists(track_folder_path):
                os.makedirs(track_folder_path)
                
            # Loop over the 5 frames of the track
            for i, frame_number in enumerate(track_bbox):
            
                cap = cv2.VideoCapture(self.video_path)
                cap.set(1, frame_number[0])
                ret, frame = cap.read()
                cap.release()
                
                # Using the bbox from the faces_id dict to crop the image
                
                s = frame_number[1]
                x = frame_number[2]
                y = frame_number[3]
                
                cs  = self.crop_scale
                bs  = s   # Detection box size
                bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
                frame = np.pad(frame, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
                my  = y + bsi  # BBox center Y
                mx  = x + bsi  # BBox center X
                face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]

                
                # Save the image
                file_name = track_id + "_" + str(i) + ".jpg"
                
                cv2.imwrite(os.path.join(track_folder_path, file_name), cv2.resize(face, (224, 224)))
    
    def cluster_tracks_face_embedding(self, track_speaking_faces, tracks):
        
        self.store_face_verification_track_images(tracks)
        
        # Model will automatically be downloaded if not present
        
        # model = insightface.model_zoo.get_model(model_file)
        # model.prepare(ctx_id=-1, det_size=(224, 224))
        # buffalo_l https://drive.google.com/file/d/1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB/view?usp=sharing
        
        # Downloading smaller model manually if want to use it (but worse performance)
        # buffalo_sc https://drive.google.com/file/d/19I-MZdctYKmVf3nu5Da3HS6KH5LBfdzG/view?usp=sharing
        
        # Check if a file naming "model.pkl" exists in the folder faces_id_path (if yes, then load the pkl file), otherwise initialize the model
        
        model_path = os.path.join(self.tracks_faces_clustering_path, 'model.pkl')
        
        if os.path.isfile(model_path):
            model = pickle.load(open(model_path, 'rb'))
        else:
            model = FaceAnalysis("buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            pickle.dump(model, open(model_path, 'wb'))
        
        model.prepare(ctx_id=0, det_size=(224, 224))
        
        # * Calculate the embeddings
        # Create a numpy array to store the embeddings
        # embeddings = np.zeros((len(tracks),512))
        embeddings = {}
        
        # sorted_files = sorted(os.listdir(model_path))
        # # Filter the files to only include the jpg files
        # sorted_files = [file for file in sorted_files if file.endswith('.jpg')]
        
        # for track_id in sorted_files:
        #     # Check if the image is a jpg file
        #     img = cv2.imread(os.path.join(faces_id_path, track_id))
        #     face = model.get(img)
        #     embedding = face[0].normed_embedding
        #     embeddings[track_id] = embedding
        #     # embeddings[i] = embedding
        
        # Go through each of the folders in self.tracks_faces_clustering_path, and for each folder, go through each of the images in the folder and calculate the embedding
        # Then, average the embeddings of the images in the folder and store the average embedding in the embeddings dict
        sorted_files = sorted(os.listdir(self.tracks_faces_clustering_path))
        # Filter the files to only include the jpg files 
        sorted_files = [file for file in sorted_files if file.isdigit()]
        

        embeddings_list = list(embeddings.values())

        # * Plotting    
        # Reduce the dimensionality of the embedding vector using PCA
        pca = PCA(n_components=2)
        embedding_2d = pca.fit_transform(embeddings_list)
        
        # Plot the 2d embeddings list (with the track_id as labels)
        fig, ax = plt.subplots(figsize=(8, 8))
        for i, track_id in enumerate(sorted_files):
            ax.scatter(embedding_2d[i, 0], embedding_2d[i, 1], label=track_id)
        ax.set_title('Face Embedding')
        plt.legend()
        plt.show()
        
        #embeddings_list = embedding_2d
        
        # * Calculate the distance matrix for debugging (to know the distance between each embedding)
        # Calculate the distance matrix
        track_dist = np.zeros((len(tracks), len(tracks)))
        for i in range(len(tracks)):
            for j in range(len(tracks)):
                # TODO: calculate cosine similarity instead of euclidean distance
                track_dist[i,j] = np.dot(embeddings_list[i], embeddings_list[j]) / (np.linalg.norm(embeddings_list[i]) * np.linalg.norm(embeddings_list[j]))
                
                # Euclidean distance
                # track_dist[i,j] = np.linalg.norm(embeddings_list[i] - embeddings_list[j])
        
        # * Clustering
        # Perform clustering with DBSCAN
        threshold_diff_person = 0.3
        dbscan = DBSCAN(eps=(1-threshold_diff_person), min_samples=2, metric='cosine')
        labels = dbscan.fit_predict(embeddings_list)

        speaking_clusters = self.parse_clusters(labels, track_speaking_faces)
                    
        return speaking_clusters
    
    # Holds the assumption that people do not change place
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
            track_avg_x[tidx] = np.mean(track)
        track_avg_y = [[] for i in range(len(tracks))]
        for tidx, track in enumerate(track_list_y):
            track_avg_y[tidx] = np.mean(track)    

        
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
        track_dist = np.zeros((len(tracks), len(tracks)))
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
        
        speaking_clusters = self.parse_clusters(labels, track_speaking_faces)

        return speaking_clusters
    
    def parse_clusters(self, labels, track_speaking_faces):
        # Print the indices of each cluster (if the label is -1, then print it as a cluster with only one index)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:
                print("Outlier:")
                outlier_indices = np.where(labels == label)[0]
                for i in outlier_indices:
                    if track_speaking_faces[i] != []:
                        print(f"\t\t{i}")
            else:
                print(f"Cluster {label}:")
                cluster_indices = np.where(labels == label)[0]
                for i in cluster_indices:
                    if track_speaking_faces[i] != []:
                        print(f"\t\t{i}")
        
            

        # Store the indices of the tracks that belong to the same cluster
        clusters = []
        for i in range(0, labels.max() + 1):
            clusters.append(np.where(labels == i)[0])

                
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