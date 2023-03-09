import pandas as pd

from src.audio.utils.time_series_features import TimeSeriesFeatures

from src.audio.utils.constants import PERMA_MODEL_TRAINING_DATA


# TODO: Create manually one CSV per day per team (create a new CSV, give every speaker a unique ID, and add the PERMA score for the day)

class PermaModel:
    def __init__(self):
        self.times_series_features = TimeSeriesFeatures()
    
    def calculate_features(self):
        
        # Read in the CSV file
        csv_path = PERMA_MODEL_TRAINING_DATA / 'test.csv'
        
        # Read the pandas dataframe and set "Speaker ID" as index
        df = pd.read_csv(csv_path)

        # Get the columns containing the emotions, and the communication patterns (each a different dataframe)
        # x -> value over time, y -> speaker ID
        # df_dominance = df[[col for col in df.columns if col.startswith('dominance')]]
        
        # TODO: make it more modular (e.g. by providing list of features in config file -> but model does only work with these features?)
        overall_df = self.times_series_features.calc_statistical_features(df)

        print("test")

        # Save values in a csv (PERMA score can then be added manually)
        # -> one csv per day containing for each speaker (9 x 4-5) ~45 input features + PERMA score
        # Speaker ID for training not necessary (only for matching to PERMA score), but for inference it is (-> to assign PERMA score to correct Person)

        # TODO: Once added all x and y (PERMA scores) to a csv, then also standardize and normalize the x values??
        
    # TODO: regarding PERMA score: standardize and normalize
    def calculate_perma_score(self):
        pass