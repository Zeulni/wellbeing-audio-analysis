import pandas as pd

from src.audio.utils.time_series_features import TimeSeriesFeatures

from src.audio.utils.constants import PERMA_MODEL_TRAINING_DATA


# TODO: Create manually one CSV per day per team (create a new CSV, give every speaker a unique ID, and add the PERMA score for the day)

class PermaModel:
    def __init__(self):
        self.times_series_features = TimeSeriesFeatures()
    
    # Data Set creation
    # Input: Team level (only for the speaker I have the informed consent - not checking for PERMA yet)
    # - Option 1: the time series data per day
    # - Option 2: the time series data for all 3 days (then the ID has to be different -> e.g. "1_1", "1_2", "1_3" for the first speaker on the first day, second day, third day)
        # team 1, speaker 1, day 1 -> 11_1, 11_2, 11_3
    # Output: 
    # - Option 1: One CSV file per team (PERMA score matching manually afterwards)
    # - Option 2: One CSV file per team including the PERMA scores (matched automatically by speaker ID, e.g. 11_1, 11_2, 11_3)
    
    # Afterwards:
    # - Concat all teams into one CSV file
    # - Standardize and normalize the data
    # - Train model
    # - Save normalization, standardication, and model function in a pickle file
    def calculate_features(self):
        
        # Read in the CSV file
        csv_path = PERMA_MODEL_TRAINING_DATA / 'test.csv'
        # csv_path = PERMA_MODEL_TRAINING_DATA / 'test_sample.csv'
        
        # Read the pandas dataframe and set "Speaker ID" as index
        df = pd.read_csv(csv_path)
        # For test_sample, use ";" as delimiter
        # df = pd.read_csv(csv_path, delimiter=';')
        
        feature_names = ['arousal', 'dominance', 'valence', 
                         'norm_num_overlaps_absolute', 'norm_num_overlaps_relative', 
                         'norm_num_turns_absolute', 'norm_num_turns_relative', 
                         'norm_speak_duration_absolute', 'norm_speak_duration_relative']
        
        # For each speaker: 9 time series x 5 features = 45 features
        short_overall_df, long_overall_df = self.times_series_features.calc_time_series_features(df, feature_names)

        print("test")
        
        # Save the short and long time series features in a csv file
        short_overall_df.to_csv(PERMA_MODEL_TRAINING_DATA / 'short_overall_df.csv')
        long_overall_df.to_csv(PERMA_MODEL_TRAINING_DATA / 'long_overall_df.csv')


        # Save values in a csv (PERMA score can then be added manually)
        # -> one csv per day containing for each speaker (9 x 4-5) ~45 input features + PERMA score
        # Speaker ID for training not necessary (only for matching to PERMA score), but for inference it is (-> to assign PERMA score to correct Person)

        # TODO: Once added all x and y (PERMA scores) to a csv, then also standardize and normalize the x values??
        # TODO: Especially for long_overall -> filter out columns with NaN or 0s
        
    # TODO: regarding PERMA score: standardize and normalize
    def calculate_perma_score(self):
        pass