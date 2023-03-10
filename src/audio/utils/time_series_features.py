import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

class TimeSeriesFeatures:
    def __init__(self):
        pass
    
    def calc_statistical_features(self, df):
        df_dominance = self.calc_ind_feature(df, 'dominance')
        df_arousal = self.calc_ind_feature(df, 'arousal')
        df_valence = self.calc_ind_feature(df, 'valence')
        df_norm_num_overlaps_absolute = self.calc_ind_feature(df, 'norm_num_overlaps_absolute')
        df_norm_num_overlaps_relative = self.calc_ind_feature(df, 'norm_num_overlaps_relative')
        df_norm_num_turns_absolute = self.calc_ind_feature(df, 'norm_num_turns_absolute')
        df_norm_num_turns_relative = self.calc_ind_feature(df, 'norm_num_turns_relative')
        df_norm_speak_duration_absolute = self.calc_ind_feature(df, 'norm_speak_duration_absolute')
        df_norm_speak_duration_relative = self.calc_ind_feature(df, 'norm_speak_duration_relative')

        # Concatenate the dataframes (per person per PERMA score ~45 features), based on Speaker ID
        overall_df = pd.concat([df_dominance, df_arousal, df_valence, df_norm_num_overlaps_absolute, df_norm_num_overlaps_relative, df_norm_num_turns_absolute, df_norm_num_turns_relative, df_norm_speak_duration_absolute, df_norm_speak_duration_relative], axis=1)
        
        return overall_df
    
    def calc_ind_feature(self, df, feature_name):
        # Calculate statistics for feature columns (e.g. dominance, valence, arousal)
        feature_cols = [col for col in df.columns if col.startswith(feature_name)]
        # feature_stats = ['mean', 'slope', 'min', 'max', 'std']

        feature_values = df[feature_cols].values
        feature_mean = np.mean(feature_values, axis=1)
        feature_slope = np.apply_along_axis(self.calc_slope, axis=1, arr=feature_values)
        feature_min = np.min(feature_values, axis=1)
        feature_max = np.max(feature_values, axis=1)
        feature_std = np.std(feature_values, axis=1)

        # Create new dataframe with calculated statistics
        # new_df = pd.DataFrame({
        #     'Speaker ID': df['Speaker ID'],
        #     'mean': feature_mean,
        #     'slope': feature_slope,
        #     'min': feature_min,
        #     'max': feature_max,
        #     'std': feature_std
        # })
        
        # The column names contain the statistic (e.g. mean) and the feature name (e.g. dominance)
        new_df = pd.DataFrame({
            'Speaker ID': df['Speaker ID'],
            'mean_' + feature_name: feature_mean,
            'slope_' + feature_name: feature_slope,
            'min_' + feature_name: feature_min,
            'max_' + feature_name: feature_max,
            'std_' + feature_name: feature_std
        })
        
        
        # Set the speaker ID as index
        new_df.set_index('Speaker ID', inplace=True)
        
        return new_df

    def calc_slope(self, feature_values):
        x = np.arange(len(feature_values))
        slope, _ = np.polyfit(x, feature_values, 1)
        return slope

    # def calculate_slope(self, feature_values):
    #     # Create a linear regression object
    #     lr = LinearRegression()

    #     # Fit a linear regression line to the feature values
    #     lr.fit(feature_values[:, :-1], feature_values[:, -1])

    #     # Get the slope of the fitted line
    #     slope = lr.coef_[0]

    #     return slope