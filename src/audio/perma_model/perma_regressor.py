from numpy import mean, std, absolute
import pickle as pkl
import matplotlib.pyplot as plt
import shap

from catboost import CatBoostRegressor, cv, Pool
import xgboost as xgb

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

from src.audio.utils.constants import PERMA_MODEL_DIR

class PermaRegressor:
    def __init__(self, data_X, data_y, database_name) -> None:
        self.data_X = data_X
        self.data_y = data_y
        self.database_name = database_name
        
    def train_model(self, multioutput_reg, param_grid, model_name):
        
        loo = LeaveOneOut()
        grid_search = GridSearchCV(multioutput_reg, param_grid, cv=loo, scoring='neg_mean_absolute_error')
        grid_search.fit(self.data_X, self.data_y)
        
        print(model_name + " Best Hyperparameters:", grid_search.best_params_)
        print(model_name + " Best Score:", -grid_search.best_score_)
        
        # Fit the MultiOutputRegressor object with the best hyperparameters
        multioutput_reg.set_params(**grid_search.best_params_)
        multioutput_reg.fit(self.data_X, self.data_y)
        
        self.plot_and_save_feature_importance(multioutput_reg, model_name)
        self.plot_and_save_shap_values(multioutput_reg, model_name)
    
        # Save models dict to pickle file
        model_file_name = self.database_name + '_' + model_name + '_perma_model.pkl'
        with open(PERMA_MODEL_DIR / model_file_name, 'wb') as f:
            pkl.dump(multioutput_reg, f)       
    
    def catboost_train(self):
        
        # Define the parameter grid to search over
        param_grid = {
            'estimator__max_depth': [3, 5, 7],
            'estimator__learning_rate': [0.1, 0.01, 0.001],
            'estimator__n_estimators': [50, 100, 200]
        }
        
        multioutput_reg = MultiOutputRegressor(CatBoostRegressor(loss_function='RMSE' ,verbose=False, save_snapshot=False, allow_writing_files=False, train_dir=str(PERMA_MODEL_DIR)))
        self.train_model(multioutput_reg, param_grid, 'catboost')    


    def xgboost_train(self):
        
        # Define the parameter grid to search over
        param_grid = {
            'estimator__max_depth': [3, 5, 7],
            'estimator__learning_rate': [0.1, 0.01, 0.001],
            'estimator__n_estimators': [50, 100, 200]
        }
        
        multioutput_reg = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror'))
        self.train_model(multioutput_reg, param_grid, 'xgboost')
        
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
        
        plt.savefig(PERMA_MODEL_DIR / f'{self.database_name}_{model_name}_feature_importance.png')
        
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
            plt.savefig(PERMA_MODEL_DIR / f'{self.database_name}_{model_name}_shap_values_{self.data_y.columns[i]}.png')
            plt.show()
            plt.clf()