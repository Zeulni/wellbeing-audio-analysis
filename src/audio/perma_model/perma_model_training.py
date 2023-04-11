import numpy as np
import pandas as pd

from src.audio.perma_model.data_scaling import DataScaling
from src.audio.perma_model.exploratory_data_analysis import ExploratoryDataAnalysis
from src.audio.perma_model.feature_reduction import FeatureReduction
from src.audio.perma_model.sample_reduction import SampleReduction
from src.audio.utils.analysis_tools import read_final_database, create_perma_results_folder, transform_test_data

from sklearn.model_selection import train_test_split

from src.audio.perma_model.perma_regressor import PermaRegressor
from src.audio.perma_model.perma_classifier import PermaClassifier

class PermaModelTraining:
    def __init__(self):
        self.data_scaling = DataScaling()
        self.exp_data_analysis = ExploratoryDataAnalysis()
        self.feature_reduction = FeatureReduction()
        self.sample_reduction = SampleReduction()
    
    def run(self): 

        # database_list = ["short_data", "long_data"]
        database_list = ["short_data"]
        
        best_param = {"short_data": {"threshold_variance": 0.04, "threshold_correlation": 0.9, "alpha_rfe": 0.01},
                      "long_data": {"threshold_variance": 0.04, "threshold_correlation": 0.9, "alpha_rfe": 0.01}}
        
        
        for database in database_list:
            # Read the data
            data = read_final_database(database)
            create_perma_results_folder()
            # Extract features and targets
            data_X_full = data.iloc[:, 5:] # features
            data_y_full = data.iloc[:, :5] # targets
            
            split_ratio = 0.20
            data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(data_X_full, data_y_full, test_size=split_ratio, random_state=1)
            data_X_train = data_X_train.reset_index(drop=True)
            data_X_test = data_X_test.reset_index(drop=True)
            data_y_train = data_y_train.reset_index(drop=True)
            data_y_test = data_y_test.reset_index(drop=True)
            
            
            # * Remove NaN columns
            # TODO: Step 1 (for data_X_test)
            data_X_train, nan_columns = self.feature_reduction.remove_nan_columns(data_X_train, database)
        
            # * Outlier Removal
            # RMSE because of higher penalty for outliers (most of the data scentered around mean -> RMSE good metric)
            # Only searches for outliers in X, outliers in PERMA (target) were already removed beforehand
            # Assumption: filter out ~3 outliers in the input data (data_X_train)
            data_X_train, data_y_train = self.sample_reduction.remove_outliers(data_X_train, data_y_train) # Perform it before scaling features
            
            # percentiles = np.linspace(0, 100, num=(5))
            # # Iterate through the lenght of the data_y_train
            # for perma_pillar in data_y_train:
            #     perma_pillar_data = data_y_train[perma_pillar]
            #     print(perma_pillar_data.value_counts().sort_index())
            #     bin_edges = np.percentile(perma_pillar_data, percentiles)
            #     bin_edges[0] = -np.inf
            #     bin_edges[-1] = np.inf
                
            #     # Print the distribution of each class in the train set
            #     print("Distribution of classes in train set for " + perma_pillar + ":")
            #     print(pd.cut(perma_pillar_data, bins=bin_edges, labels=False).value_counts().sort_index())
            
            # * EDA
            # self.exp_data_analysis.plot_correlation_matrices_feature_names(data_X_train, data_y_train)
            # self.exp_data_analysis.plot_correlation_matrices_feature_types(data_X_train, data_y)
            # self.exp_data_analysis.plot_pairplots_feature_names(data_X_train, data_y_train)
            # self.exp_data_analysis.print_stats(data_X_train)
            # self.exp_data_analysis.plot_perma_pillars(data_y_train)

            # * Scaling
            # TODO: Step 2 (for data_X_test and data_y_test)
            data_y_train, normalize_scaler = self.data_scaling.normalize_targets(data_y_train)       
            # data_y_train = self.scaling_data.standardize_targets(data_y)     
            data_X_train, gaussian_columns, gaussian_feature_scaler, non_gaussian_columns, nongaussian_feature_scaler = self.data_scaling.scale_features(data_X_train, database)
            
            
            # * Feature Selection 
            # TODO: Step 3 (for data_X_test)
            data_X_train = self.feature_reduction.variance_thresholding(data_X_train, best_param[database])
            data_X_train, correlated_features = self.feature_reduction.correlation_clustering(data_X_train, data_y_train, best_param[database])
            # data_X_train = self.feature_reduction.perform_pca(data_X_train, database, 10)
            # data_X_train, perma_feature_list = self.feature_reduction.select_features_mutual_info(data_X_train, data_y_train, database, 6)
            data_X_train, perma_feature_list, unique_features = self.feature_reduction.recursive_feature_elimination(data_X_train, data_y_train, database, best_param[database])
            
            # Filter correlated features based on the remaining columns in data_X_train
            # tbd
            
            # self.feature_reduction.finding_best_k_mutual_info(data_X_train, data_y_train)
            # data_X_train = self.feature_reduction.select_features_mutual_info(data_X_train, data_y_train, database, 2)
            # data_X_train = self.feature_reduction.select_features_regression(data_X_train, data_y_train, database)
            # data_X_train = self.feature_reduction.perform_pca(data_X_train, database, 8)
            
            # self.exp_data_analysis.plot_pairplot_final_features(data_X_train, data_y_train, feature_importance_dict)
            # self.exp_data_analysis.plot_correlations_with_target(data_X_train, data_y_train)
            
            # TODO: rename short:small and long:large
            # TODO: TODO: Test for Long dataset the function "X_filtered = select_features(X, y)" for each pillar
            
            data_X_test, data_y_test = transform_test_data(data_X_test, data_y_test, nan_columns, normalize_scaler, gaussian_columns, gaussian_feature_scaler, non_gaussian_columns, nongaussian_feature_scaler, unique_features)
            
            # * Model Training
            perma_regressor = PermaRegressor(data_X_train, data_X_test, data_y_train, data_y_test, perma_feature_list, database)
            perma_regressor.train_multiple_models_per_pillar()
            
            for number_classes in range(2,5):
                # Create a copy of data_X_train, data_X_test, data_y_train, data_y_test
                data_X_train_copy = data_X_train.copy()
                data_X_test_copy = data_X_test.copy()
                data_y_train_copy = data_y_train.copy()
                data_y_test_copy = data_y_test.copy()
                
                perma_classifier = PermaClassifier(data_X_train_copy, data_X_test_copy, data_y_train_copy, data_y_test_copy, perma_feature_list, database, number_classes)
                perma_classifier.train_multiple_models_per_pillar()
            
            
            print("--------------------------------------------------")