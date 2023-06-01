import os
import pandas as pd

from src.audio.perma_model.data_scaling import DataScaling
from src.audio.perma_model.exploratory_data_analysis import ExploratoryDataAnalysis
from src.audio.perma_model.feature_reduction import FeatureReduction
from src.audio.perma_model.sample_reduction import SampleReduction
from src.audio.utils.analysis_tools import read_final_database, create_dataset_for_sdm_plots, create_perma_results_folder, transform_test_data

from sklearn.model_selection import train_test_split

from src.audio.perma_model.perma_regressor import PermaRegressor
from src.audio.perma_model.perma_classifier import PermaClassifier

from src.audio.utils.constants import PERMA_MODEL_RESULTS_DIR

# Grayscale color palette:
#000000
#333333
#666666
#999999
#CCCCCC

class PermaModelTraining:
    def __init__(self):
        self.data_scaling = DataScaling()
        self.exp_data_analysis = ExploratoryDataAnalysis()
        self.feature_reduction = FeatureReduction()
        self.sample_reduction = SampleReduction()
    
    def run(self): 

        # database_list = ["small_data", "large_data"]
        database_list = ["small_data"]
        
        best_param = {"small_data": {"threshold_variance": 0.04, "threshold_correlation": 0.9, "alpha_rfe": 0.01, "number_classes": 2},
                        "large_data": {"threshold_variance": 0.04, "threshold_correlation": 0.9, "alpha_rfe": 0.01, "number_classes": 3}}
        
        
        for database in database_list:            
            
            # Read the data
            # Old naming convention: If database = "short_data" -> rename it to "small_data", If database = "long_data" -> rename it to "large_data"
            if "small" in database:
                database_fileread = database.replace("small", "short")
            elif "large" in database:
                database_fileread = database.replace("large", "long")
            data = read_final_database(database_fileread)
            # create_dataset_for_sdm_plots(database_fileread)
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
            data_X_train, nan_columns = self.feature_reduction.remove_nan_columns(data_X_train, database)
        
            # * Outlier Removal
            # RMSE because of higher penalty for outliers (most of the data scentered around mean -> RMSE good metric)
            # Only searches for outliers in X, outliers in PERMA (target) were already removed beforehand
            # Assumption: filter out ~3 outliers in the input data (data_X_train)
            data_X_train, data_y_train = self.sample_reduction.remove_outliers(data_X_train, data_y_train) # Perform it before scaling features
            
            
            # * EDA
            # self.exp_data_analysis.plot_correlation_matrices_feature_names(data_X_train, data_y_train)
            # self.exp_data_analysis.plot_correlation_matrices_feature_types(data_X_train, data_y)
            # self.exp_data_analysis.plot_pairplots_feature_names(data_X_train, data_y_train)
            # self.exp_data_analysis.print_stats(data_X_train)
            # self.exp_data_analysis.plot_perma_pillars(data_y_train)

            # * Scaling
            data_y_train, normalize_scaler = self.data_scaling.normalize_targets(data_y_train)       
            # data_y_train = self.scaling_data.standardize_targets(data_y)     
            data_X_train, gaussian_columns, gaussian_feature_scaler, non_gaussian_columns, nongaussian_feature_scaler = self.data_scaling.scale_features(data_X_train, database)
            
            
            # * Feature Selection 
            data_X_train = self.feature_reduction.variance_thresholding(data_X_train, best_param[database])
            data_X_train, correlated_features = self.feature_reduction.correlation_clustering(data_X_train, data_y_train, best_param[database])
            
            data_X_train_file = os.path.join(PERMA_MODEL_RESULTS_DIR, database + "_reduced_data_X.pkl")
            perma_feature_list_file = os.path.join(PERMA_MODEL_RESULTS_DIR, database + "_perma_feature_list.pkl")
            unique_features_file = os.path.join(PERMA_MODEL_RESULTS_DIR, database + "_reduced_data_X_features.pkl")
            
            # If all files exist, load them
            if os.path.exists(data_X_train_file) and os.path.exists(perma_feature_list_file) and os.path.exists(unique_features_file):
                data_X_train = pd.read_pickle(data_X_train_file)
                perma_feature_list = pd.read_pickle(perma_feature_list_file)
                unique_features = pd.read_pickle(unique_features_file)
            else:
                data_X_train, perma_feature_list, unique_features = self.feature_reduction.recursive_feature_elimination(data_X_train, data_y_train, database, best_param[database])
            
            
            # self.feature_reduction.finding_best_k_mutual_info(data_X_train, data_y_train)
            # data_X_train = self.feature_reduction.select_features_mutual_info(data_X_train, data_y_train, database, 2)
            # data_X_train = self.feature_reduction.select_features_regression(data_X_train, data_y_train, database)
            # data_X_train = self.feature_reduction.perform_pca(data_X_train, database, 8)
            
            # self.exp_data_analysis.plot_pairplot_final_features(data_X_train, data_y_train, feature_importance_dict)
            # self.exp_data_analysis.plot_correlations_with_target(data_X_train, data_y_train, perma_feature_list)
            
            data_X_test, data_y_test = transform_test_data(data_X_test, data_y_test, nan_columns, normalize_scaler, gaussian_columns, gaussian_feature_scaler, non_gaussian_columns, nongaussian_feature_scaler, unique_features)
            
            # * Model Training
            # perma_regressor = PermaRegressor(data_X_train, data_X_test, data_y_train, data_y_test, perma_feature_list, database)
            # perma_regressor.train_multiple_models_per_pillar()
            # perma_regressor.calc_rel_baseline_comp()
            
            # for number_classes in range(2,5):
            #     # Create a copy of data_X_train, data_X_test, data_y_train, data_y_test
            #     data_X_train_copy = data_X_train.copy()
            #     data_X_test_copy = data_X_test.copy()
            #     data_y_train_copy = data_y_train.copy()
            #     data_y_test_copy = data_y_test.copy()
                
            #     perma_classifier = PermaClassifier(data_X_train_copy, data_X_test_copy, data_y_train_copy, data_y_test_copy, perma_feature_list, database, number_classes)
            #     perma_classifier.train_multiple_models_per_pillar()
            
            perma_classifier = PermaClassifier(data_X_train, data_X_test, data_y_train, data_y_test, perma_feature_list, database, best_param[database]["number_classes"])
            perma_classifier.train_multiple_models_per_pillar()
    
            # perma_classifier.plot_rel_baseline_comp_n_classes()
            
            
            print("--------------------------------------------------")