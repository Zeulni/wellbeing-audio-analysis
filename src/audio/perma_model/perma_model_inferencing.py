import os
import pandas as pd
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
from PIL import Image

from src.audio.utils.time_series_features import TimeSeriesFeatures

from src.audio.utils.constants import FEATURE_NAMES
from src.audio.utils.constants import PERMA_MODEL_RESULTS_DIR


class PermaModelInferencing:
    def __init__(self, csv_path, save_path, faces_id_path, perma_scale, logger) -> None:
        self.csv_path = csv_path
        self.save_path = save_path
        self.faces_id_path = faces_id_path
        self.perma_scale = perma_scale
        self.logger = logger
        self.times_series_features = TimeSeriesFeatures()
        
            
    def load_data(self):
        # * Load the csv file
        if os.path.isfile(self.csv_path):
            self.time_series_df = pd.read_csv(self.csv_path)
        else:
            raise FileNotFoundError(f"{self.csv_path} does not exist.")
        
        # * Load the columns and scaler for the gaussian features
        with open(PERMA_MODEL_RESULTS_DIR / "small_data_gaussian_columns.pkl", 'rb') as f:
            self.gaussian_columns = pkl.load(f)
        
        with open(PERMA_MODEL_RESULTS_DIR / "small_data_feature_std_scaler.pkl", 'rb') as f:
            self.feature_std_scaler = pkl.load(f)
            
        # * Load the columns and scaler for the non-gaussian features
        with open(PERMA_MODEL_RESULTS_DIR / "small_data_non_gaussian_columns.pkl", 'rb') as f:
            self.non_gaussian_columns = pkl.load(f)
            
        with open(PERMA_MODEL_RESULTS_DIR / "small_data_feature_robust_scaler.pkl", 'rb') as f:
            self.feature_robust_scaler = pkl.load(f)
            
        # * Load the columns selected by the feature selection algorithm    
        with open(PERMA_MODEL_RESULTS_DIR / "small_data_perma_feature_list.pkl", 'rb') as f:
            self.selected_features = pkl.load(f)
            
        # * Load the best REGRESSION models
        with open(PERMA_MODEL_RESULTS_DIR / "regression_small_data_best_models.pkl", 'rb') as f:
            self.perma_regression_models = pkl.load(f)
            
        # * Load the best CLASSIFICATION models
        with open(PERMA_MODEL_RESULTS_DIR / "classification_n2_small_data_best_models.pkl", 'rb') as f:
            self.perma_classification_models = pkl.load(f)
        
        # * Load the scaler for the regression models  
        with open(PERMA_MODEL_RESULTS_DIR / "perma_norm_scaler.pkl", 'rb') as f:
            self.perma_scaler = pkl.load(f)
    
    def run(self):
        
        # Load all necessary data
        self.load_data()
    
        # Calculate the time series features (long vs. short)
        feature_df, long_feature_df = self.times_series_features.calc_time_series_features(self.time_series_df, FEATURE_NAMES)
        
        
        # Scale the features
        gaussian_feature_array = self.feature_std_scaler.transform(feature_df[self.gaussian_columns])
        gaussian_feature_df = pd.DataFrame(gaussian_feature_array, index=feature_df.index, columns=self.gaussian_columns)
        non_gaussian_feature_array = self.feature_robust_scaler.transform(feature_df[self.non_gaussian_columns])
        non_gaussian_feature_df = pd.DataFrame(non_gaussian_feature_array, index=feature_df.index, columns=self.non_gaussian_columns)
        feature_df = pd.concat([gaussian_feature_df, non_gaussian_feature_df], axis=1)
        
        # Transform to dataframe again (based on index and colum names of selftime_series_df)
        # feature_df = pd.DataFrame(feature_array, index=feature_df.index, columns=feature_df.columns)
        
        # Select only columns based on feature selection algorithm from the dataframe 
        # feature_df = feature_df[self.selected_features]
        feature_df_P = feature_df[self.selected_features[0]]
        feature_df_E = feature_df[self.selected_features[1]]
        feature_df_R = feature_df[self.selected_features[2]]
        feature_df_M = feature_df[self.selected_features[3]]
        feature_df_A = feature_df[self.selected_features[4]]
        
        # * Run the regression model    
        perma_regression_scores_P = np.expand_dims(self.perma_regression_models[0].predict(feature_df_P), axis=1)
        perma_regression_scores_E = np.expand_dims(self.perma_regression_models[1].predict(feature_df_E), axis=1)
        perma_regression_scores_R = np.expand_dims(self.perma_regression_models[2].predict(feature_df_R), axis=1)
        perma_regression_scores_M = np.expand_dims(self.perma_regression_models[3].predict(feature_df_M), axis=1)
        perma_regression_scores_A = np.expand_dims(self.perma_regression_models[4].predict(feature_df_A), axis=1)
        
        # Concatenate the results
        perma_regression_scores = np.concatenate([perma_regression_scores_P, perma_regression_scores_E, perma_regression_scores_R, perma_regression_scores_M, perma_regression_scores_A], axis=1)

        # Undo scaling of PERMA scores   
        if self.perma_scale == "default":
            perma_regression_scores = self.perma_scaler.inverse_transform(perma_regression_scores)
            # Clip the values to be between 0 and 7 (definition of our PERMA scores)
            perma_regression_scores = perma_regression_scores.clip(0, 7)
        elif self.perma_scale == "norm":
            perma_regression_scores = perma_regression_scores.clip(0, 1)
        else:
            raise ValueError(f"perma_scale {self.perma_scale} is not supported.")
        
        # Saves the regression results to a csv file
        perma_regression_df = pd.DataFrame(perma_regression_scores, columns=["P", "E", "R", "M", "A"],index=feature_df.index) 
        csv_regression_path = os.path.join(self.save_path, "perma_regression_scores.csv")        
        perma_regression_df.to_csv(csv_regression_path)
        self.logger.log(f"Saved regression PERMA scores to {csv_regression_path}")
        print("Saved regression PERMA scores to: ", csv_regression_path)
        
        # * Run the classification model
        perma_classification_scores_P = np.expand_dims(self.perma_classification_models[0].predict(feature_df_P), axis=1)
        perma_classification_scores_E = np.expand_dims(self.perma_classification_models[1].predict(feature_df_E), axis=1)
        perma_classification_scores_R = np.expand_dims(self.perma_classification_models[2].predict(feature_df_R), axis=1)
        perma_classification_scores_M = np.expand_dims(self.perma_classification_models[3].predict(feature_df_M), axis=1)
        perma_classification_scores_A = np.expand_dims(self.perma_classification_models[4].predict(feature_df_A), axis=1)
        
        # Concatenate the results
        perma_classification_scores = np.concatenate([perma_classification_scores_P, perma_classification_scores_E, perma_classification_scores_R, perma_classification_scores_M, perma_classification_scores_A], axis=1)
        # Change all 0 to "low" and 1 to "high"
        perma_classification_scores = np.where(perma_classification_scores == 0, "low", "high")
        
        # Saves the classification results to a csv file
        perma_classification_df = pd.DataFrame(perma_classification_scores, columns=["P", "E", "R", "M", "A"],index=feature_df.index)
        csv_classification_path = os.path.join(self.save_path, "perma_classification_scores.csv")
        perma_classification_df.to_csv(csv_classification_path)
        self.logger.log(f"Saved classification PERMA scores to {csv_classification_path}")
        print("Saved classification PERMA scores to: ", csv_classification_path)

        # Visualize the results
        self.plot_perma_result(perma_regression_df)
    
    def plot_perma_result(self, perma_df):

        # Set up the figure and subplots
        fig, axes = plt.subplots(nrows=1, ncols=len(perma_df), figsize=(15, 8),
                                subplot_kw=dict(polar=True))

        # Iterate over each row of the data and create a spider chart for each row
        for ax, (_, row) in zip(axes, perma_df.iterrows()):
            angles = [n / float(len(row)) * 2 * np.pi for n in range(len(row))]
            angles += angles[:1]
            values = row.tolist() + row.tolist()[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid')
            ax.fill(angles, values, alpha=0.3)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(row.index, fontsize=10)
            if self.perma_scale == "default":
                ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
                ax.set_ylim([0, 7])
            elif self.perma_scale == "norm":
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_ylim([0.0, 1.0])
            ax.set_title(row.name, fontsize=12)
            ax.grid(True)



            speaker_id = str(row.name).split()[-1]
            file_name = [f for f in os.listdir(self.faces_id_path) if speaker_id in f][0]
            img = mpimg.imread(os.path.join(self.faces_id_path, file_name))

            # Define the bbox of the current ax to use it to place the image
            Bbox = ax.get_position()

            # Create new axes for the image at the top of the current axes
            ax_image = fig.add_axes([Bbox.x0, Bbox.y1+0.1, Bbox.width, 0.2], anchor='S') # Change 0.1 to change the height of the image's axes

            # Hide the spines, ticks, and labels of the image's axes
            ax_image.axis("off")

            # Display the image in the new axes
            ax_image.imshow(img)


        # Adjust the layout and save the figure
        #fig.tight_layout()
        plt.savefig(os.path.join(self.save_path, "perma_spider_charts.png"), dpi=500)
        plt.show()