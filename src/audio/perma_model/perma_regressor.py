import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import shap

from catboost import CatBoostRegressor
import xgboost as xgb

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import make_scorer

from math import sqrt

from src.audio.utils.constants import PERMA_MODEL_RESULTS_DIR

class PermaRegressor:
    def __init__(self, data_X, data_y, database_name) -> None:
        self.data_X = data_X
        self.data_y = data_y
        self.database_name = database_name
        
    def train_model(self, multioutput_reg, model_name):
        # Define the parameter grid to search over
        # param_grid = {
        #     'estimator__max_depth': [3, 5, 7],
        #     'estimator__learning_rate': [0.1, 0.01, 0.001],
        #     'estimator__n_estimators': [50, 100, 200]
        # }
        
        param_grid = {
            'estimator__max_depth': [5],
            'estimator__learning_rate': [0.01],
            'estimator__n_estimators': [100]
        }
        
        # param_grid = {
        #     'estimator__max_depth': [3, 5, 7],
        #     'estimator__learning_rate': [0.1, 0.01],
        #     'estimator__n_estimators': [100, 200]
        # }
        
        
        loo = LeaveOneOut()
        # TODO: change to RMSE?
        scoring = 'root_mean_squared_error'
        rmse_scorer = make_scorer(lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
        # scoring = 'mean_absolute_error'
        # msa_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        grid_search = GridSearchCV(multioutput_reg, param_grid, cv=loo, scoring=rmse_scorer)
        grid_search.fit(self.data_X, self.data_y)
        
        print(model_name + " Best Hyperparameters:", grid_search.best_params_)
        print(model_name + " Best Score " + scoring + ": ", -grid_search.best_score_)
        
        # Fit the MultiOutputRegressor object with the best hyperparameters
        multioutput_reg.set_params(**grid_search.best_params_)
        multioutput_reg.fit(self.data_X, self.data_y)
        
        # Print baseline RMSE and MAE
        self.calc_baseline(model_name)
        
        # self.plot_and_save_feature_importance(multioutput_reg, model_name)
        # self.plot_and_save_shap_values(multioutput_reg, model_name)
    
        # Save models dict to pickle file
        model_file_name = self.database_name + '_' + model_name + '_perma_model.pkl'
        with open(PERMA_MODEL_RESULTS_DIR / model_file_name, 'wb') as f:
            pkl.dump(multioutput_reg, f)

    def calc_baseline(self, model_name):
        
        # Calculate the mean of the PERMA scores
        mean_perma_scores = self.data_y.mean(axis=0)
        
        y_pred_baseline = np.tile(mean_perma_scores, (len(self.data_y), 1))
        
        # Calculate the RMSE of the baseline model
        baseline_rmse = np.sqrt(mean_squared_error(self.data_y, y_pred_baseline))
        
        # Calculate the MAE of the baseline model
        baseline_mae = mean_absolute_error(self.data_y, y_pred_baseline)
        
        print(model_name + " Baseline RMSE:", baseline_rmse)
        print(model_name + " Baseline MAE:", baseline_mae)
    
    def catboost_train(self):
        
        multioutput_reg = MultiOutputRegressor(CatBoostRegressor(loss_function='RMSE' ,verbose=False, save_snapshot=False, allow_writing_files=False, train_dir=str(PERMA_MODEL_RESULTS_DIR)))
        self.train_model(multioutput_reg, 'catboost')    


    def xgboost_train(self):
        
        # Define the parameter grid to search over
        # param_grid = {
        #     'estimator__max_depth': [3, 5, 7],
        #     'estimator__learning_rate': [0.1, 0.01, 0.001],
        #     'estimator__n_estimators': [50, 100, 200]
        # }
        
        multioutput_reg = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror'))
        self.train_model(multioutput_reg, 'xgboost')
        
    def plot_and_save_feature_importance(self, regr, model_name):
        
        # Close all current plots
        plt.close('all')
        
        # Define the size of the plot
        plt.figure(figsize=(16, 8))
        
        # Loop over every estimator and plot the feature importance as a subplot
        for i, estimator in enumerate(regr.estimators_):
            plt.subplot(2, 3, i+1)
            sorted_feature_importance = estimator.feature_importances_.argsort()
            plt.barh(self.data_X.columns[sorted_feature_importance], 
                    estimator.feature_importances_[sorted_feature_importance], 
                    color='turquoise')
            plt.xlabel(model_name + " Feature Importance")
            plt.title(self.data_y.columns[i])
            plt.tight_layout()
        
        plt.savefig(PERMA_MODEL_RESULTS_DIR / f'{self.database_name}_{model_name}_feature_importance.png')
        
        plt.show()
        plt.clf()
        
    def plot_and_save_shap_values(self, regr, model_name):        
    
        # Only plot shap values for the top 5 important features for each estimator
        for i, estimator in enumerate(regr.estimators_):
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(self.data_X)
            
            show_last_n_features = 5
            sorted_feature_importance = estimator.feature_importances_.argsort()[-show_last_n_features:]
            columns_shap_values = self.data_X.columns[sorted_feature_importance]
            shap_values = shap_values[:, sorted_feature_importance]

            shap.summary_plot(shap_values, self.data_X[columns_shap_values], feature_names = columns_shap_values, show=False)
            # plt.xlabel("SHAP Values")
            plt.title(self.data_y.columns[i])
            plt.savefig(PERMA_MODEL_RESULTS_DIR / f'{self.database_name}_{model_name}_shap_values_{self.data_y.columns[i]}.png')
            plt.show()
            plt.clf()