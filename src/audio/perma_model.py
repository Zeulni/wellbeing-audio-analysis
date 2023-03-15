import os
import glob
import pandas as pd
from pathlib import Path

from src.audio.utils.time_series_features import TimeSeriesFeatures

# from src.audio.utils.constants import PERMA_MODEL_TRAINING_DATA
from src.audio.utils.constants import VIDEOS_DIR


# Input: The CSV files per team (time series)
# Output: The calculated features per team (from the time series) concatenated for each day, including the corresponding PERMA value

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
    def calculate_features(self, team):
        
        # Read in the CSV file
        # csv_path = PERMA_MODEL_TRAINING_DATA / 'test.csv'
        # csv_path = PERMA_MODEL_TRAINING_DATA / 'test_sample.csv'
        
        # Read the pandas dataframe and set "Speaker ID" as index
        # df = pd.read_csv(csv_path)
        # For test_sample, use ";" as delimiter
        # df = pd.read_csv(csv_path, delimiter=';')
        
        feature_names = ['arousal', 'dominance', 'valence', 
                         'norm_num_overlaps_absolute', 'norm_num_overlaps_relative', 
                         'norm_num_turns_absolute', 'norm_num_turns_relative', 
                         'norm_speak_duration_absolute', 'norm_speak_duration_relative']
                
        
        # Save the short and long time series features in a csv file
        # short_overall_df.to_csv(PERMA_MODEL_TRAINING_DATA / 'short_overall_df.csv')
        # long_overall_df.to_csv(PERMA_MODEL_TRAINING_DATA / 'long_overall_df.csv')

        team_folder = str(VIDEOS_DIR / team)

        # Loop over the subdirectories (i.e., the day folders)
        for day_folder in sorted(os.listdir(team_folder)):
            day_path = os.path.join(team_folder, day_folder)
            
            if os.path.isdir(day_path):
                
                team_day_features = pd.DataFrame()
                
                # Loop over the video files within each day folder (the order is important, so the time line for one day is correct)
                folder_names = [f for f in os.listdir(day_path) if os.path.isdir(os.path.join(day_path, f))]
                sorted_folder_names = sorted(folder_names, key=self.custom_sort)
                
                # Concatenate the features for each day in the right time sequence
                for clip_folder in sorted_folder_names:
                    
                    clip_folder_path = os.path.join(day_path, clip_folder)
                    # Get the file path of a csv file in the clip_folder
                    csv_file = glob.glob(os.path.join(clip_folder_path, "*.csv"))
    
                    if not csv_file:
                        print("No csv file found for path: " + clip_folder_path)
                        continue
                    else:
                        csv_path = csv_file[0]
                        
                    df = pd.read_csv(csv_path)
                    df.set_index('Speaker ID', inplace=True)
                    
                    # If team_day_features is empty, just copy the df
                    if team_day_features.empty:
                        team_day_features = df.copy()
                    else:
                        team_day_features = pd.concat([team_day_features, df], axis=1, join='outer')

                team_day_features = team_day_features.reset_index().drop_duplicates(subset='Speaker ID', keep='last')          
                
                # print(team_day_features)
                team_day_features.to_csv(VIDEOS_DIR / 'test_team_day_features.csv')

                short_feature_df, long_feature_df = self.times_series_features.calc_time_series_features(team_day_features, feature_names)
                
                self.merge_with_perma(short_feature_df, "short_overall_df", day_folder, team_folder, team)
                self.merge_with_perma(long_feature_df, "long_overall_df", day_folder, team_folder, team)
                

        # Save values in a csv (PERMA score can then be added manually)
        # -> one csv per day containing for each speaker (9 x 4-5) ~45 input features + PERMA score
        # Speaker ID for training not necessary (only for matching to PERMA score), but for inference it is (-> to assign PERMA score to correct Person)

        # TODO: Once added all x and y (PERMA scores) to a csv, then also standardize and normalize the x values??
        # TODO: Especially for long_overall -> filter out columns with NaN or 0s
        
    # TODO: regarding PERMA score: standardize and normalize
    def calculate_perma_score(self):
        pass
    
    def custom_sort(self, folder_name):
        # Split the folder name into parts
        parts = folder_name.split('_')
        
        # Extract the clip number and frame numbers
        clip_num = int(parts[1])
        start_frame = int(parts[2])
        
        # Return a tuple to define the sorting order
        return (clip_num, start_frame)
    
    def merge_with_perma(self, short_feature_df, file_name, day_folder, team_folder, team): 
        perma_path = VIDEOS_DIR / "perma_scores_dataset.csv"
        df_perma = pd.read_csv(perma_path)
        
        # One day is e.g. "2023-01-10" -> extract the 10
        day = int(day_folder.split("-")[2])
        
        # In short_feature_df_team, the column "Speaker ID", which is also the index, contains the alias 
        # Filter the df_perma for the day 
        # Then do a left join on the alias, where df_perma is the left df which is the basis and the short_feature_df_team is the right df which is the one that is added
        df_perma_filtered = df_perma[(df_perma['Day'] == day)]
        # print(df_perma_filtered)

        # Then, perform a left join on the Alias column
        merged_df = pd.merge(df_perma_filtered, short_feature_df, left_on='Alias', right_index=True, how='left')
        # Only drop the row if all columns (apart from the ones form the left df) are NaN
        merged_df = merged_df.dropna(how='all', subset=short_feature_df.columns)
        # print(merged_df)
                        
        # Save files with day_folder in the name
        merged_df.to_csv(Path(team_folder) / f'{team}_{file_name}_{day_folder}.csv')
