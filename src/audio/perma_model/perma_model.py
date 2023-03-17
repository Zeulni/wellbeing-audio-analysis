import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor


from src.audio.utils.constants import PERMA_MODEL

class PermaModel:
    def __init__(self) -> pd:
        pass
    
    def read_dataframe(self, folder) -> None:
        # Read the csvs as dataframes (just run everytime again instead of checking if csv is available -> always up to date)
        data_folder = PERMA_MODEL / folder
        
        # Read all the csvs in the data folder as dataframes and append them in one dataframe
        data = pd.DataFrame()
        
        # Create a list of all csv files in data_folder and sort them
        csv_files = sorted([file for file in data_folder.glob("*.csv")])
        for file in csv_files:
            data = pd.concat([data, pd.read_csv(file)], axis=0)
            
        # Remove the rows "Unnamed: 0", "E-Mail-Adresse", "Alias", "First Name", "Last Name/Surname", "Day"
        data = data.drop(["Unnamed: 0", "E-Mail-Adresse", "Alias", "First Name", "Last Name/Surname", "Day"], axis=1)
        
        # Reset the index
        data = data.reset_index(drop=True)

        # Save the dataframe as csv
        data.to_csv(os.path.join(PERMA_MODEL, folder + ".csv"))
        
        return data
    
    # LOF works well for high dimensional data
    def detect_outliers(self, data_X, data_y) -> pd:
        
        # Extract the features and target variables
        X = data_X.values # features
        y = data_y.values # targets

        # Set LOF parameters for feature outlier detection
        n_neighbors_f = 20
        contamination_f = 0.05

        # Set LOF parameters for target outlier detection
        n_neighbors_t = 20
        contamination_t = 0.05

        # Fit the LOF model for feature outlier detection
        lof_f = LocalOutlierFactor(n_neighbors=n_neighbors_f, contamination=contamination_f)
        lof_f.fit(X)

        # Predict the outlier scores for feature outlier detection
        scores_f = lof_f.negative_outlier_factor_

        # Determine the threshold for outlier detection for feature outlier detection
        threshold_f = np.percentile(scores_f, 100 * contamination_f)

        # Identify the feature outliers
        outliers_f = X[scores_f < threshold_f]

        # Fit the LOF models for target outlier detection
        lof_t = []
        for i in range(y.shape[1]):
            lof = LocalOutlierFactor(n_neighbors=n_neighbors_t, contamination=contamination_t)
            lof.fit(y[:, i].reshape(-1, 1))
            lof_t.append(lof)

        # Predict the outlier scores for target outlier detection
        scores_t = np.zeros_like(y)
        for i in range(y.shape[1]):
            scores_t[:, i] = lof_t[i].negative_outlier_factor_.reshape(-1)

        # Determine the threshold for outlier detection for target outlier detection
        threshold_t = np.percentile(scores_t, 100 * contamination_t, axis=0)

        # Identify the target outliers
        # TODO: shape is different than outliers_f -> cause for problem with printing?
        outliers_t = y[scores_t < threshold_t]

        # Print the number of outliers and their indices for feature outlier detection
        print('Number of feature outliers:', len(outliers_f))
        print('Feature outlier indices:', [i for i, x in enumerate(X) if any((x == y).all() for y in outliers_f)])

        # Print the number of outliers and their indices for target outlier detection
        print('Number of target outliers:', len(outliers_t))
        print('Target outlier indices:', [i for i, x in enumerate(y) if any((x == y).all() for y in outliers_t)])
    
    def plot_perma_pillars(self, data_y) -> None:
        
        # # Plot each target variable as a scatter plot
        # for col in data_y.columns:
        #     plt.scatter(data_y.index, data_y[col], label=col)

        # plt.legend()
        # plt.show()

        # Plot each target variable as a box plot
        data_y.boxplot()
        plt.show()
    
    def handle_missing_values(self, data, filename) -> pd:
        
        # TODO: any or all?
        # Stats in long data: 4656 rows have at least one NaN value, 3105 rows have all NaN values
        
        # Store all columns where all values are NaN
        columns_with_nan = []
        for col in data.columns:
            if data[col].isnull().any():
                columns_with_nan.append(col)
                
        # Print the columns with NaN values
        print("Columns with NaN values: ", len(columns_with_nan))
        
        # TODO: open: how to handle missing values? Drop them? Impute them?
        # TODO: store deleted columns, needed for inference
        # Remove the columns with all NaN values
        data = data.drop(columns_with_nan, axis=1)
        
        # Store the columns with NaN values in a pickle file
        with open(os.path.join(PERMA_MODEL, filename + "_columns_with_nan.pkl"), "wb") as f:
            pkl.dump(columns_with_nan, f)
        
        # Overwrite existing csv with updated dataframe
        data.to_csv(os.path.join(PERMA_MODEL, filename + ".csv"))
        
        return data

    def standardize_features(self, data_X) -> pd:
        
        scaler = StandardScaler()
        scaler.fit(data_X)
        
        # Save the scaler as pickle file (to use it for inference later on)        
        with open(os.path.join(PERMA_MODEL, "data_short_scaler.pkl"), "wb") as f:
            pkl.dump(scaler, f)

        # fit and transform the DataFrame using the scaler
        array_standardized = scaler.transform(data_X)
        
        # Repalce the original columns with the standardized columns
        data_X_standardized = pd.DataFrame(array_standardized, columns=data_X.columns)
        
        return data_X_standardized
    
    def run(self): 

        short_data = self.read_dataframe("short_data")
        short_data_X = short_data.iloc[:, 5:] # features
        short_data_y = short_data.iloc[:, :5] # targets
        
        # long_data = self.read_dataframe("long_data")
        # long_data_X = long_data.iloc[:, 5:] # features
        # long_data_y = long_data.iloc[:, :5] # targets
        
        # TODO: open: how exactly to perform outlier detection? Only on PERMA scores? how to handle the outliers?
        # self.detect_outliers(short_data_X, short_data_y)
    
        # self.plot_perma_pillars(short_data_y)
        
        short_data = self.handle_missing_values(short_data, "short_data")
        # long_data = self.handle_missing_values(long_data, "long_data")
        
        short_data_X = self.standardize_features(short_data_X)
        
        print("test")