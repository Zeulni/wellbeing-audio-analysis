import os
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from insightface.app import FaceAnalysis

class ClusterTracks:
    def __init__(self) -> None:
        pass
    
    def cluster_tracks_face_embedding(self, track_speaking_faces, faces_id_path, tracks):
        # Model will automatically be downloaded if not present
        
        # Only download the model once, then load the model
        # model_name = 'buffalo_l'
        # model_file = f'{model_name}-0000.params'
        # model_url = f'https://github.com/deepinsight/insightface/models/{model_name}/{model_file}'

        # insightface.utils.download(model_url, model_file)
        
        # model = insightface.model_zoo.get_model(model_file)
        # model.prepare(ctx_id=-1, det_size=(224, 224))
        # buffalo_l https://drive.google.com/file/d/1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB/view?usp=sharing
        
        # Downloading smaller model manually if want to use it (but worse performance)
        # buffalo_sc https://drive.google.com/file/d/19I-MZdctYKmVf3nu5Da3HS6KH5LBfdzG/view?usp=sharing
        
        model = FaceAnalysis("buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        model.prepare(ctx_id=0, det_size=(224, 224))
        
        # * Calculate the embeddings
        # Create a numpy array to store the embeddings
        # embeddings = np.zeros((len(tracks),512))
        embeddings = {}
        
        sorted_files = sorted(os.listdir(faces_id_path))
        # Filter the files to only include the jpg files
        sorted_files = [file for file in sorted_files if file.endswith('.jpg')]
        
        for track_id in sorted_files:
            # Check if the image is a jpg file
            img = cv2.imread(os.path.join(faces_id_path, track_id))
            face = model.get(img)
            embedding = face[0].normed_embedding
            embeddings[track_id] = embedding
            # embeddings[i] = embedding

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
        
        embeddings_list = embedding_2d
        
        # * Calculate the distance matrix for debugging (to know the distance between each embedding)
        # Calculate the distance matrix
        track_dist = np.zeros((len(tracks), len(tracks)))
        for i in range(len(tracks)):
            for j in range(len(tracks)):
                # track_dist[i,j] = np.linalg.norm(embeddings_list[i] - embeddings_list[j])
                track_dist[i,j] = math.sqrt((embeddings_list[i,0] - embeddings_list[j,0])**2 + (embeddings_list[i,1] - embeddings_list[j,1])**2)
        
        # * Clustering
        # Perform clustering with DBSCAN
        threshold_diff_person = 0.3
        dbscan = DBSCAN(eps=threshold_diff_person, min_samples=2)
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