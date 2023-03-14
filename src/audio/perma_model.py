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
    def calculate_features(self):
        
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

        team = 'team_13'
        team_folder = str(VIDEOS_DIR / team)

        # Loop over the subdirectories (i.e., the day folders)
        for day_folder in sorted(os.listdir(team_folder)):
            day_path = os.path.join(team_folder, day_folder)
            
            if os.path.isdir(day_path):
                
                short_feature_df_team = pd.DataFrame()
                long_feature_df_team = pd.DataFrame()
                
                # Loop over the video files within each day folder (the order is important, so the time line for one day is correct)
                folder_names = [f for f in os.listdir(day_path) if os.path.isdir(os.path.join(day_path, f))]
                sorted_folder_names = sorted(folder_names, key=self.custom_sort)
                for clip_folder in sorted_folder_names:
                    # Check that the file is an MP4
                    
                    clip_folder_path = os.path.join(day_path, clip_folder)
                    
                    if os.path.isdir(clip_folder_path):
                        # Get the file path of a csv file in the clip_folder
                        
                        csv_file = glob.glob(os.path.join(clip_folder_path, "*.csv"))
        
                        if not csv_file:
                            raise Exception("No csv file found for path: " + clip_folder_path)
                        else:
                            csv_path = csv_file[0]
                            
                        df = pd.read_csv(csv_path)

                        # TODO: I have to first concatenate them per day and then perform 1x per day feature extraction!
                            
                        # TODO: bring back long features
                        short_feature_df, long_feature_df = self.times_series_features.calc_time_series_features(df, feature_names)
                            
                        short_feature_df_team = pd.concat([short_feature_df_team, short_feature_df], axis=1)
                        long_feature_df_team = pd.concat([long_feature_df_team, long_feature_df], axis=1)
                        
                        # TODO: Remove this to part where features per day are calculated
                        perma_path = VIDEOS_DIR / "perma_scores_dataset.csv"
                        df_perma = pd.read_csv(perma_path)
                        
                        # One day is e.g. "2023-01-10" -> extract the 10
                        day = int(day_folder.split("-")[2])
                        
                        # In short_feature_df_team, the column "Speaker ID", which is also the index, contains the alias 
                        # Filter the df_perma for the day and the alias (stored in Speaker ID). The alias in df_perma is stored in the column "Alias"
                        # Then do a left join on the alias, where df_perma is the left df which is the basis and the short_feature_df_team is the right df which is the one that is added
                        # Code:
                        alias = short_feature_df_team.index[0] # get the alias from the index of short_feature_df_team
                        df_perma_filtered = df_perma[(df_perma['Day'] == day)]
                        print(df_perma_filtered)

                        # Then, perform a left join on the Alias column
                        merged_df = pd.merge(df_perma_filtered, short_feature_df_team, left_on='Alias', right_index=True, how='left')
                        # Only drop the row if all columns (apart from the ones form the left df) are NaN
                        merged_df = merged_df.dropna(how='all', subset=short_feature_df_team.columns)
                        print(merged_df)
                        
                        # When the alias is not found in the df_perma_filtered, then delete that row and don't save it in merged_df
                            
                        # Save df_perma
                        merged_df.to_csv(VIDEOS_DIR / 'test_overall_dataset.csv')
                            
                        print("test")

                # short_feature_df_team.to_csv(team_folder / 'short_overall_df.csv')
                # long_feature_df_team.to_csv(team_folder / 'long_overall_df.csv')
                
                perma_path = VIDEOS_DIR / "perma_scores_dataset.csv"
                df_perma = pd.read_csv(perma_path)
                
                # One day is e.g. "2023-01-10" -> extract the 10
                day = day_folder.split("-")[2]
                
                # In short_feature_df_team, the speaker ID is the alias
                # Filter the df_perma for the day and the alias
                # Then to a left join on the alias, where short_feature_df_team is the left df (store it in a new dataframe)
                # Code:
                df_perma[df_perma['day'] == day].merge(short_feature_df_team, how='left', left_on='alias', right_on='Speaker ID')


                
                # Save files with day_folder in the name
                short_feature_df_team.to_csv(Path(team_folder) / f'short_overall_df_{day_folder}.csv')
                long_feature_df_team.to_csv(Path(team_folder) / f'long_overall_df_{day_folder}.csv')

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