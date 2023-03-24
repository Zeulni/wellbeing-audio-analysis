import os
import pickle as pkl
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from src.audio.utils.constants import PERMA_MODEL_RESULTS_DIR

class ScalingData():
    def __init__(self) -> None:
        pass
    
    def standardize_features(self, data_X, database) -> pd:
                
        scaler = StandardScaler()
        scaler.fit(data_X)
        
        # Save the scaler as pickle file (to use it for inference later on)        
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, database + "_feature_std_scaler.pkl"), "wb") as f:
            pkl.dump(scaler, f)

        # fit and transform the DataFrame using the scaler
        array_standardized = scaler.transform(data_X)
        
        # Repalce the original columns with the standardized columns
        data_X_standardized = pd.DataFrame(array_standardized, columns=data_X.columns)
        
        return data_X_standardized
    
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

        return data_y_normalized
    
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