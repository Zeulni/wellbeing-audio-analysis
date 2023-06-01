import numpy as np
import pandas as pd
from tsfresh import extract_features

class TimeSeriesFeatures:
    def __init__(self):
        pass
    
    def calc_time_series_features(self, df, feature_names):        
        # Make it more modular (e.g. by providing list of features)
        small_overall_df = pd.DataFrame()
        large_overall_df = pd.DataFrame()
        for feature_name in feature_names:
            small_feature_df = self.calc_ind_feature(df, feature_name)
            small_overall_df = pd.concat([small_overall_df, small_feature_df], axis=1)
        
            # large_feature_df = self.calc_tsfresh_features(df, feature_name)
            # large_overall_df = pd.concat([large_overall_df, large_feature_df], axis=1)
            
        return small_overall_df, large_overall_df
    
    def calc_ind_feature(self, df, feature_name):
                
        # Columns that contain feature name
        feature_cols = [col for col in df.columns if feature_name in col]

        feature_values = df[feature_cols].values
        
        # Exclude all NaN values (where I don't have a data point)
        feature_mean = np.nanmean(feature_values, axis=1)
        feature_slope = np.apply_along_axis(self.calc_slope, axis=1, arr=feature_values)
        feature_min = np.nanmin(feature_values, axis=1)
        feature_max = np.nanmax(feature_values, axis=1)
        feature_std = np.nanstd(feature_values, axis=1)
        feature_var = np.nanvar(feature_values, axis=1)
        feature_median = np.nanmedian(feature_values, axis=1)
        feature_q25 = np.nanquantile(feature_values, 0.25, axis=1)
        feature_q75 = np.nanquantile(feature_values, 0.75, axis=1)
        
        # The column names contain the statistic (e.g. mean) and the feature name (e.g. dominance)        
        new_df = pd.DataFrame({
            'Speaker ID': df['Speaker ID'],
            feature_name + '_mean': feature_mean,
            feature_name + '_slope': feature_slope,
            feature_name + '_min': feature_min,
            feature_name + '_max': feature_max,
            feature_name + '_std': feature_std,
            feature_name + '_var': feature_var,
            feature_name + '_median': feature_median,
            feature_name + '_q25': feature_q25,
            feature_name + '_q75': feature_q75
        })  
        
        # Set the speaker ID as index
        new_df.set_index('Speaker ID', inplace=True)
        
        return new_df

    def calc_slope(self, feature_values):

        
        # Just remove NaN values (e.g. mit be the case that a person in the meetings in not in the middle one -> so just concatenate first and last meeting)
        feature_values_nonnan = feature_values[~np.isnan(feature_values)]
        
        # If length is 1, then the slope is 0
        if len(feature_values_nonnan) == 1:
            return 0
        
        x = np.arange(len(feature_values_nonnan))
        slope, _ = np.polyfit(x, feature_values_nonnan, 1)
        
        
        # Tested version to insert mean, but doesn't make sense
        # nan_mask = np.isnan(feature_values)
        # mean_value = np.nanmean(feature_values)
        # feature_values_imputed = np.where(nan_mask, mean_value, feature_values)

        # x = np.arange(len(feature_values_imputed))
        # slope, _ = np.polyfit(x, feature_values_imputed, 1)
                        
        return slope
    
    
    def calc_tsfresh_features(self, df, feature_name):

        # extract colums that contain the feature name
        value_vars = [col for col in df.columns if feature_name in col]

        # use the melt function to transform the dataframe to a timeseries format
        id_vars = ['Speaker ID']
        df_timeseries = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='time', value_name=feature_name)

        # sort the dataframe by speaker id and time
        df_timeseries = df_timeseries.sort_values(by=['Speaker ID', 'time']).reset_index(drop=True)
        
        # Remove rows with NaN values
        df_timeseries = df_timeseries.dropna()
 
        df_features = extract_features(df_timeseries, column_id='Speaker ID', column_sort='time')
        
        # Name the index "Speaker ID"
        df_features.index.name = 'Speaker ID'
        
        return df_features