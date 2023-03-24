import os
import glob
import pandas as pd
from pathlib import Path

from src.audio.utils.time_series_features import TimeSeriesFeatures

# from src.audio.utils.constants import PERMA_MODEL_TRAINING_DATA
from src.audio.utils.constants import VIDEOS_DIR
from src.audio.utils.constants import FEATURE_NAMES
from src.audio.utils.constants import PERMA_MODEL_DIR

class CalculateTimeSeriesFeatures:
    def __init__(self):
        self.times_series_features = TimeSeriesFeatures()
    
    # Data Set creation for training the PERMA model
    def run(self, team):

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
                for clip_id, clip_folder in enumerate(sorted_folder_names):
                    
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
                    
                    # Add the clip ID in front of every column name (e.g. "arousal" -> "0_arousal")
                    df = df.add_prefix(str(clip_id) + '_')
                    
                    # If team_day_features is empty, just copy the df
                    if team_day_features.empty:
                        team_day_features = df.copy()
                    else:
                        team_day_features = pd.concat([team_day_features, df], axis=1, join='outer')

                team_day_features = team_day_features.reset_index().drop_duplicates(subset='Speaker ID', keep='last')
                
                # Make sure all Speaker IDs are strings
                team_day_features['Speaker ID'] = team_day_features['Speaker ID'].astype(str)          
                
                # print(team_day_features)
                # team_day_features.to_csv(VIDEOS_DIR / 'test_team_day_features.csv')

                short_feature_df, long_feature_df = self.times_series_features.calc_time_series_features(team_day_features, FEATURE_NAMES)
                
                self.merge_with_perma(short_feature_df, "short_overall_df", day_folder, team_folder, team)
                self.merge_with_perma(long_feature_df, "long_overall_df", day_folder, team_folder, team)

    
    def custom_sort(self, folder_name):
        # Split the folder name into parts
        parts = folder_name.split('_')
        
        # Extract the clip number and frame numbers
        clip_num = int(parts[1])
        start_frame = int(parts[2])
        
        # Return a tuple to define the sorting order
        return (clip_num, start_frame)
    
    def merge_with_perma(self, short_feature_df, file_name, day_folder, team_folder, team): 
        perma_path = PERMA_MODEL_DIR / "perma_scores_dataset.csv"
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
