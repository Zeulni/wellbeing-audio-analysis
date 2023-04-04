import os
import pickle as pkl
import pandas as pd
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from src.audio.utils.constants import PERMA_MODEL_RESULTS_DIR

class DataScaling():
    def __init__(self) -> None:
        pass
    
    def normalize_targets(self, data_y) -> pd:

        columns = ["P", "E", "R", "M", "A"]
        
        scaler = MinMaxScaler()

        scaler.fit(data_y[columns])
        
        # Save the scaler as pickle file    
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, "perma_norm_scaler.pkl"), "wb") as f:
            pkl.dump(scaler, f)

        # fit and transform the DataFrame using the scaler
        array_normalized = scaler.transform(data_y[columns])
        
        # Repalce the original columns with the normalized columns
        data_y_normalized = data_y.copy()
        data_y_normalized[columns] = array_normalized

        return data_y_normalized, scaler
    
    def standardize_targets(self, data_y) -> pd:

        columns = ["P", "E", "R", "M", "A"]
        
        scaler = StandardScaler()

        scaler.fit(data_y[columns])
        
        # Save the scaler as pickle file    
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, "perma_std_scaler.pkl"), "wb") as f:
            pkl.dump(scaler, f)

        # fit and transform the DataFrame using the scaler
        array_standardized = scaler.transform(data_y[columns])
        
        # Repalce the original columns with the standardized columns
        data_y_standardized = data_y.copy()
        data_y_standardized[columns] = array_standardized

        return data_y_standardized
    
    def standardize_features(self, data_X, database, columns) -> pd:
                
        data_X = data_X[columns]
                
        scaler = StandardScaler()
        scaler.fit(data_X)
        
        # Save the scaler as pickle file (to use it for inference later on)        
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, database + "_feature_std_scaler.pkl"), "wb") as f:
            pkl.dump(scaler, f)

        # fit and transform the DataFrame using the scaler
        array_standardized = scaler.transform(data_X)
        
        # Repalce the original columns with the standardized columns
        data_X_standardized = pd.DataFrame(array_standardized, columns=columns)
        
        return data_X_standardized, scaler
    
    def robust_scale_features(self, data_X, database, columns) -> pd:
        
        data_X = data_X[columns]
                
        scaler = RobustScaler()
        scaler.fit(data_X)
        
        # Save the scaler as pickle file (to use it for inference later on)        
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, database + "_feature_robust_scaler.pkl"), "wb") as f:
            pkl.dump(scaler, f)

        # fit and transform the DataFrame using the scaler
        array_robust_scaled = scaler.transform(data_X)
        
        # Repalce the original columns with the robust scaled columns
        data_X_robust_scaled = pd.DataFrame(array_robust_scaled, columns=columns)
        
        return data_X_robust_scaled, scaler
    
    def determine_gaussian_columns(self, data_X, database) -> list:
        # Determine the columns with a Gaussian distribution
        gaussian_columns = []
        
        for col in data_X.columns:
            stat, p = shapiro(data_X[col])
            if p > 0.05:
                gaussian_columns.append(col)
                
        non_gaussian_columns = [col for col in data_X.columns if col not in gaussian_columns]
        
        # Save the columns with a Gaussian distribution as pickle file (to use it for inference later on)
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, database + "_gaussian_columns.pkl"), "wb") as f:
            pkl.dump(gaussian_columns, f)
            
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, database + "_non_gaussian_columns.pkl"), "wb") as f:
            pkl.dump(non_gaussian_columns, f)
        
        return gaussian_columns, non_gaussian_columns
    
    def scale_features(self, data_X, database) -> pd:
        
        gaussian_columns, non_gaussian_columns = self.determine_gaussian_columns(data_X, database)
        
        # Gaussian columns: scale using the standard scaler
        data_X_standardized, gaussian_feature_scaler = self.standardize_features(data_X, database, gaussian_columns)
        
        # Non-Gaussian columns: scale using the robust scaler
        data_X_robust_scaled, nongaussian_feature_scaler = self.robust_scale_features(data_X, database, non_gaussian_columns)
        
        # Concatenate the standardized and robust scaled columns
        data_X_scaled = pd.concat([data_X_standardized, data_X_robust_scaled], axis=1)
        
        return data_X_scaled, gaussian_columns, gaussian_feature_scaler, non_gaussian_columns, nongaussian_feature_scaler