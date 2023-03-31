import os
import pandas as pd
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
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
        
        # * Load the csv file
        if os.path.isfile(self.csv_path):
            self.time_series_df = pd.read_csv(self.csv_path)
        else:
            raise FileNotFoundError(f"{self.csv_path} does not exist.")
        
        # * Load the columns and scaler for the gaussian features
        with open(PERMA_MODEL_RESULTS_DIR / "short_data_gaussian_columns.pkl", 'rb') as f:
            self.gaussian_columns = pkl.load(f)
        
        with open(PERMA_MODEL_RESULTS_DIR / "short_data_feature_std_scaler.pkl", 'rb') as f:
            self.feature_std_scaler = pkl.load(f)
            
        # * Load the columns and scaler for the non-gaussian features
        with open(PERMA_MODEL_RESULTS_DIR / "short_data_non_gaussian_columns.pkl", 'rb') as f:
            self.non_gaussian_columns = pkl.load(f)
            
        with open(PERMA_MODEL_RESULTS_DIR / "short_data_feature_robust_scaler.pkl", 'rb') as f:
            self.feature_robust_scaler = pkl.load(f)
            
        # * Load the columns selected by the feature selection algorithm
        with open(PERMA_MODEL_RESULTS_DIR / "short_data_selected_features.pkl", 'rb') as f:
            self.selected_features = pkl.load(f)
            
        # * Load the best model: Lasso regression
        with open(PERMA_MODEL_RESULTS_DIR / "short_data_lasso_perma_model.pkl", 'rb') as f:
            self.perma_model = pkl.load(f)
            
        with open(PERMA_MODEL_RESULTS_DIR / "perma_norm_scaler.pkl", 'rb') as f:
            self.perma_scaler = pkl.load(f)
    
    def run(self):
        
        # TODO: bring long feature calculation back (comment out for now)
        # Calculate the time series features (long vs. short)
        feature_df, long_feature_df = self.times_series_features.calc_time_series_features(self.time_series_df, FEATURE_NAMES)
        
        # Debugging
        # feature_df.to_csv(os.path.join(self.save_path, "feature_df.csv"))
        
        # Scale the features
        gaussian_feature_array = self.feature_std_scaler.transform(feature_df[self.gaussian_columns])
        gaussian_feature_df = pd.DataFrame(gaussian_feature_array, index=feature_df.index, columns=self.gaussian_columns)
        non_gaussian_feature_array = self.feature_robust_scaler.transform(feature_df[self.non_gaussian_columns])
        non_gaussian_feature_df = pd.DataFrame(non_gaussian_feature_array, index=feature_df.index, columns=self.non_gaussian_columns)
        feature_df = pd.concat([gaussian_feature_df, non_gaussian_feature_df], axis=1)
        
        # Transform to dataframe again (based on index and colum names of selftime_series_df)
        # feature_df = pd.DataFrame(feature_array, index=feature_df.index, columns=feature_df.columns)
        
        # Select only columns based on feature selection algorithm from the dataframe 
        feature_df = feature_df[self.selected_features]
        
        # Run the regression model    
        perma_scores = self.perma_model.predict(feature_df)

        # Undo scaling of PERMA scores   
        if self.perma_scale == "default":
            perma_scores = self.perma_scaler.inverse_transform(perma_scores)
            # Clip the values to be between 0 and 7 (definition of our PERMA scores)
            perma_scores = perma_scores.clip(0, 7)
        elif self.perma_scale == "norm":
            perma_scores = perma_scores.clip(0, 1)
        else:
            raise ValueError(f"perma_scale {self.perma_scale} is not supported.")
        
        # Saves the results to a csv file
        perma_df = pd.DataFrame(perma_scores, columns=["P", "E", "R", "M", "A"],index=feature_df.index) 
        csv_path = os.path.join(self.save_path, "perma_scores.csv")        
        perma_df.to_csv(csv_path)
        self.logger.log(f"Saved perma scores to {csv_path}")
        print("Saved perma scores to: ", csv_path)

        # Visualize the results (filling bar chart)
        # TODO: assign to correct person and plot one perma chart per person
        self.plot_perma_result(perma_df)
        
        
    
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

            # Load the image for the speaker and add it to the figure
            # TODO: implement naming algo with "__speakername"
            speaker_id = str(row.name).split()[-1]
            # file_name = os.path.join(self.faces_id_path, f"{speaker_id}.jpg")
            # The filename starts with speaker_id, but could follow by "__speakername"
            file_name = [f for f in os.listdir(self.faces_id_path) if f.startswith(speaker_id)][0]
            img = Image.open(os.path.join(self.faces_id_path, file_name))
            # img = img.resize((100, 100))
            fig.figimage(img, xo=ax.bbox.x0+100, yo=ax.bbox.y1-200, alpha=1)


        # Adjust the layout and save the figure
        #fig.tight_layout()
        plt.savefig(os.path.join(self.save_path, "perma_spider_charts.png"))
        plt.show()