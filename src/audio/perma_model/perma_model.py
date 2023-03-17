import os

import pandas as pd
import numpy as np

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
    
    def detect_outliers(self, data) -> pd:
        
        # Extract the features and target variables
        X = data.iloc[:, :-5].values # features
        y = data.iloc[:, -5:].values # targets

        # Set LOF parameters
        n_neighbors = 20 # number of neighbors to consider
        contamination = 0.05 # percentage of outliers expected

        # Fit the LOF model
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        lof.fit(X)

        # Predict the outlier scores
        scores = lof.negative_outlier_factor_

        # Determine the threshold for outlier detection
        threshold = np.percentile(scores, 100 * contamination)

        # Identify the outliers
        outliers = X[scores < threshold]

        # Print the number of outliers and their indices
        print('Number of outliers:', len(outliers))
        print('Outlier indices:', [i for i, x in enumerate(X) if any((x == y).all() for y in outliers)])
    
    def run(self): 

        short_data = self.read_dataframe("short_data")
        self.detect_outliers(short_data)
        
        print("test")