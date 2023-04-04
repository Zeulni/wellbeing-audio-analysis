from src.audio.perma_model.data_scaling import DataScaling
from src.audio.perma_model.exploratory_data_analysis import ExploratoryDataAnalysis
from src.audio.perma_model.feature_reduction import FeatureReduction
from src.audio.perma_model.sample_reduction import SampleReduction
from src.audio.utils.analysis_tools import read_final_database, create_perma_results_folder

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
            data_X = data.iloc[:, 5:] # features
            data_y = data.iloc[:, :5] # targets
            
            data_X = self.feature_reduction.remove_nan_columns(data_X, database)
           
            
            # RMSE because of higher penalty for outliers (most of the data scentered around mean -> RMSE good metric)
            # Only searches for outliers in X, outliers in PERMA (target) were already removed beforehand
            # Assumption: filter out ~3 outliers in the input data (data_X)
            data_X, data_y = self.sample_reduction.remove_outliers(data_X, data_y) # Perform it before scaling features
            
            # * EDA
            # self.exp_data_analysis.plot_correlation_matrices_feature_names(data_X, data_y)
            # self.exp_data_analysis.plot_correlation_matrices_feature_types(data_X, data_y)
            # self.exp_data_analysis.plot_pairplots_feature_names(data_X, data_y)
            # self.exp_data_analysis.print_stats(data_X)
            
            # * Scaling
            data_y = self.data_scaling.normalize_targets(data_y)       
            # data_y = self.scaling_data.standardize_targets(data_y)     
            data_X = self.data_scaling.scale_features(data_X, database)
            
            
            # * Feature Selection
            data_X = self.feature_reduction.variance_thresholding(data_X, best_param[database])
            data_X, correlated_features = self.feature_reduction.correlation_thresholding(data_X, data_y, best_param[database])
            # data_X = self.feature_reduction.perform_pca(data_X, database, 10)
            # data_X, perma_feature_list = self.feature_reduction.select_features_mutual_info(data_X, data_y, database, 6)
            data_X, perma_feature_list = self.feature_reduction.recursive_feature_elimination(data_X, data_y, database, best_param[database])
            
            # Filter correlated features based on the remaining columns in data_X
            # tbd
            
            # self.feature_reduction.finding_best_k_mutual_info(data_X, data_y)
            # data_X = self.feature_reduction.select_features_mutual_info(data_X, data_y, database, 2)
            # data_X = self.feature_reduction.select_features_regression(data_X, data_y, database)
            # data_X = self.feature_reduction.perform_pca(data_X, database, 8)
            
            # self.exp_data_analysis.plot_pairplot_final_features(data_X, data_y, feature_importance_dict)
            # self.exp_data_analysis.plot_perma_pillars(data_y)
            # self.exp_data_analysis.plot_correlations_with_target(data_X, data_y)
            
            
            # * Model Training
            perma_regressor = PermaRegressor(data_X, data_y, perma_feature_list, database)
            # perma_regressor.train_multiple_models_per_pillar()
            # perma_regressor.lasso_train()
            # perma_regressor.catboost_train()
            # perma_regressor.xgboost_train()
            
            # TODO: use different feature selection for classification (especially rcv?)
            data_X_train, data_X_test, data_y_train, data_y_test = perma_regressor.get_dataset()
            perma_classifier = PermaClassifier(data_X_train, data_X_test, data_y_train, data_y_test, perma_feature_list, database)
            perma_classifier.train_multiple_models_per_pillar()
            
            
            print("--------------------------------------------------")