from src.audio.perma_model.scaling_data import ScalingData
from src.audio.perma_model.exploratory_data_analysis import ExploratoryDataAnalysis
from src.audio.perma_model.feature_reduction import FeatureReduction
from src.audio.perma_model.sample_reduction import SampleReduction
from src.audio.utils.analysis_tools import read_final_database, create_perma_results_folder

from src.audio.perma_model.perma_regressor import PermaRegressor

class PermaModelTraining:
    def __init__(self):
        self.scaling_data = ScalingData()
        self.exp_data_analysis = ExploratoryDataAnalysis()
        self.feature_reduction = FeatureReduction()
        self.sample_reduction = SampleReduction()
    
    def run(self): 

        # database_list = ["short_data", "long_data"]
        database_list = ["short_data"]
        
        best_param = {"short_data": {"threshold_variance": 0.4, "threshold_correlation": 0.9, "alpha_rfe": 0.01, "n_estimators": 200, "max_depth": 3, "learning_rate": 0.01, "model": "catboost"},
                      "long_data": {"threshold_variance": 0.4, "threshold_correlation": 0.9, "n_estimators": 200, "max_depth": 3, "learning_rate": 0.01, "model": "catboost"}}
        
        for database in database_list:
            # Read the data
            data = read_final_database(database)
            create_perma_results_folder()
            # Extract features and targets
            data_X = data.iloc[:, 5:] # features
            data_y = data.iloc[:, :5] # targets
            
            data_X = self.feature_reduction.remove_nan_columns(data_X, database)
            # TODO: occams razor -> choose simpler model with e.g. depth 3 and 100 estimators
            
            # TODO: problem: as avg. of PERMA scores at 5.5 and std deviation low -> also all predictions in that range
            # TODO: solution: use normalization instead of standardization, and then have final scores between 0 and 1?
           
            
            # TODO: write in Overleaf the entire process (how calculated PERMA scores, how outlier, how scaled, why used which scaling,...)
            # TODO: RMSE because of higher penalty for outliers (most of the data scentered around mean -> RMSE good metric)
            # Only searches for outliers in X, outliers in PERMA (target) were already removed beforehand
            # Assumption: filter out ~3 outliers in the input data (data_X)
            data_X, data_y = self.sample_reduction.remove_outliers(data_X, data_y) # Perform it before scaling features
            
            # * EDA
            self.exp_data_analysis.plot_correlation_matrices_feature_names(data_X, data_y)
            # self.exp_data_analysis.plot_correlation_matrices_feature_types(data_X, data_y)
            # self.exp_data_analysis.plot_pairplots_feature_names(data_X, data_y)
            # self.exp_data_analysis.print_stats(data_X)
            
            # * Scaling
            data_y = self.scaling_data.normalize_targets(data_y)       
            # data_y = self.scaling_data.standardize_targets(data_y)     
            data_X = self.scaling_data.scale_features(data_X, database)
            
            # * Feature Selection
            # TODO: if take 3 step selection method, just store final features in a list and then select them
            data_X = self.feature_reduction.variance_thresholding(data_X, best_param[database])
            data_X = self.feature_reduction.correlation_thresholding(data_X, best_param[database])
            data_X = self.feature_reduction.recursive_feature_elimination(data_X, data_y, database, best_param[database])
            # self.feature_reduction.finding_best_k_mutual_info(data_X, data_y)
            # data_X = self.feature_reduction.select_features_mutual_info(data_X, data_y, database)
            # data_X = self.feature_reduction.select_features_regression(data_X, data_y, database)
            # data_X = self.feature_reduction.perform_pca(data_X, database)
            
            # self.exp_data_analysis.plot_pairplot_final_features(data_X, data_y, feature_importance_dict)
            # self.exp_data_analysis.plot_perma_pillars(data_y)
            # TODO: get all plots back in
            self.exp_data_analysis.plot_correlations_with_target(data_X, data_y)
            
            
            # * Model Training
            perma_regressor = PermaRegressor(data_X, data_y, database)
            perma_regressor.catboost_train()
            # perma_regressor.xgboost_train()
            print("--------------------------------------------------")