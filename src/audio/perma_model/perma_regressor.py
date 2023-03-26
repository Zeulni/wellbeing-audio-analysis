import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import shap

from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold


from math import sqrt

from src.audio.utils.constants import PERMA_MODEL_RESULTS_DIR

class PermaRegressor:
    def __init__(self, data_X, data_y, database_name) -> None:
        # self.data_X = data_X
        # self.data_y = data_y
        
        # Divide into train and test set (shuffle=True)
        self.data_X_train, self.data_X_test, self.data_y_train, self.data_y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
        
        self.database_name = database_name
        
    def train_model(self, multioutput_reg_model, model_name, param_grid):        
        
        # TODO: loo back?
        loo = LeaveOneOut() 
        rmse = "neg_root_mean_squared_error"
        grid_search = GridSearchCV(multioutput_reg_model, param_grid, cv=loo, scoring=rmse, verbose=0, n_jobs=-1, refit=rmse)
        grid_search.fit(self.data_X_train, self.data_y_train)
        
        print(model_name + " Best Hyperparameters:", grid_search.best_params_)
        print(model_name + " Best Score " + rmse + ": ", round(-grid_search.best_score_,3))
        
        # Fit the MultiOutputRegressor object with the best hyperparameters
        # multioutput_reg_model.set_params(**grid_search.best_params_)
        # multioutput_reg_model.fit(self.data_X, self.data_y)
        
        multioutput_reg_model = grid_search.best_estimator_
        
        
        # Print baseline RMSE and MAE
        self.calc_baseline(multioutput_reg_model, model_name)
        
        self.plot_and_save_feature_importance(multioutput_reg_model, model_name)
        # self.plot_and_save_shap_values(multioutput_reg_model, model_name)
    
        # Save models dict to pickle file
        model_file_name = self.database_name + '_' + model_name + '_perma_model.pkl'
        with open(PERMA_MODEL_RESULTS_DIR / model_file_name, 'wb') as f:
            pkl.dump(multioutput_reg_model, f)
            
    def train_ind_models(self, reg_model, model_name, param_grid):        
        
        models = []
        for i in range(self.data_y_train.shape[1]):
            models.append(reg_model)
        

        
        best_params = []
        best_scores = []
        for i in range(self.data_y_train.shape[1]):
            model = models[i]
            # scorer = ["neg_root_mean_squared_error", "r2"]
            scoring = "neg_root_mean_squared_error"
            # scoring = 'neg_mean_absolute_error'
            # scoring = "neg_mean_squared_error"
            # scoring = "r2"
            # rmse_scorer = make_scorer(lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
            # cv_factor = int(len(self.data_y_train) / 2)
            # cv = KFold(n_splits=cv_factor, shuffle=True, random_state=42)
            cv = LeaveOneOut() 
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, verbose=0, n_jobs=-1, refit=scoring)
            grid_search.fit(self.data_X_train, self.data_y_train.iloc[:, i])
            # best_alpha = grid_search.best_params_['alpha']
            # models.append(Lasso(alpha=best_alpha))
            # models_trained.append(grid_search.best_estimator_)
            models[i] = grid_search.best_estimator_
            best_params.append(grid_search.best_params_)
            # Train model with best alpha
            # models[i].fit(self.data_X, self.data_y.iloc[:, i])
            best_scores.append(-grid_search.best_score_)
            
        # Print the sum and mean of the best scores
        print(model_name + " Mean Best Score " + scoring + ": ", np.mean(best_scores))
        print(model_name + " Best Params: ", best_params)
        
        multioutput_reg_model = MultiOutputRegressor(Lasso())
        
        # Set estimators_ attribute of MultiOutputRegressor object
        multioutput_reg_model.estimators_ = models
        
        # Print baseline RMSE and MAE
        self.calc_baseline_comparison(multioutput_reg_model, model_name)
        
        self.plot_and_save_feature_importance(multioutput_reg_model, model_name)
        # self.plot_and_save_shap_values(multioutput_reg_model, model_name)
    
        # Save models dict to pickle file
        model_file_name = self.database_name + '_' + model_name + '_perma_model.pkl'
        with open(PERMA_MODEL_RESULTS_DIR / model_file_name, 'wb') as f:
            pkl.dump(multioutput_reg_model, f)

    def calc_baseline_comparison(self, multioutput_reg_model, model_name):
        
        # TODO: Baseline based on train set mean!?
        # Calculate the mean of the PERMA scores
        mean_perma_scores = self.data_y_train.mean(axis=0)
        
        y_pred_baseline = np.tile(mean_perma_scores, (len(self.data_y_test), 1))
        
        # Calculate the RMSE of the baseline model
        baseline_rmse = np.sqrt(mean_squared_error(self.data_y_test, y_pred_baseline))
        
        # Calculate the R2 score of the baseline model
        baseline_r2 = r2_score(self.data_y_test, y_pred_baseline)
        
        # Calculate the MAE of the baseline model
        baseline_mae = mean_absolute_error(self.data_y_test, y_pred_baseline)
        
        print(model_name + " Test Set Baseline r2:", round(baseline_r2, 3))
        print(model_name + " Test Set Baseline RMSE:", round(baseline_rmse,3))
        print (model_name + " Test Set Baseline MAE:", round(baseline_mae,3))
        
        # Using the entire dataset as test set (with multioutput_reg_model) to comput R2, RMSE, MAE
        y_pred = multioutput_reg_model.predict(self.data_X_test)
        r2 = r2_score(self.data_y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.data_y_test, y_pred))
        mae = mean_absolute_error(self.data_y_test, y_pred)
        
        print(model_name + " Test Set Prediction r2:", round(r2,3))
        print(model_name + " Test Set Prediction RMSE:", round(rmse,3))
        print(model_name + " Test Set Prediction MAE:", round(mae,3))
        
    
    def catboost_train(self):
        
        # Define the parameter grid to search over
        # param_grid = {
        #     'estimator__max_depth': [3, 5, 7],
        #     'estimator__learning_rate': [0.1, 0.01, 0.001],
        #     'estimator__n_estimators': [50, 100, 200]
        # }
        
        # param_grid = {
        #     'max_depth': [5],
        #     'learning_rate': [0.01],
        #     'n_estimators': [100]
        # }
        
        # param_grid = {
        #     'estimator__max_depth': [3, 5, 7],
        #     'estimator__learning_rate': [0.1, 0.01],
        #     'estimator__n_estimators': [100, 200]
        # }
        
        param_grid_ind = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [100, 200]
        }
        
        multioutput_reg_model = MultiOutputRegressor(CatBoostRegressor(loss_function='RMSE' ,verbose=False, save_snapshot=False, allow_writing_files=False, train_dir=str(PERMA_MODEL_RESULTS_DIR)))
        # self.train_model(multioutput_reg_model, 'catboost', param_grid)
        reg_model = CatBoostRegressor(loss_function='RMSE' ,verbose=False, save_snapshot=False, allow_writing_files=False, train_dir=str(PERMA_MODEL_RESULTS_DIR))
        self.train_ind_models(reg_model, 'catboost', param_grid_ind)

    def xgboost_train(self):
        
        # Define the parameter grid to search over
        # param_grid = {
        #     'estimator__max_depth': [3, 5, 7],
        #     'estimator__learning_rate': [0.1, 0.01],
        #     'estimator__n_estimators': [100, 200]
        # }
        
        param_grid_ind = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [100, 200]
        }
        
        # multioutput_reg_model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror'))
        # self.train_model(multioutput_reg_model, 'xgboost', param_grid)
        
        reg_model = xgb.XGBRegressor(objective='reg:squarederror')
        self.train_ind_models(reg_model, 'xgboost', param_grid_ind)
        
    def lasso_train(self):
        
        # Define the parameter grid to search over
        # param_grid = {
        #     'alpha': [0.001, 0.01, 0.1]
        # }
        param_grid = {
            'alpha': [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
        }
        

        
        reg_model = Lasso()
        self.train_ind_models(reg_model, 'lasso', param_grid)
        
    def plot_and_save_feature_importance(self, regr, model_name):
        
        # Close all current plots
        plt.close('all')
        
        # Define the size of the plot
        plt.figure(figsize=(16, 8))
            
        # Loop over every estimator and plot the feature importance as a subplot
        for i, estimator in enumerate(regr.estimators_):
            plt.subplot(2, 3, i+1)
            if model_name == 'catboost' or model_name == 'xgboost':
                sorted_feature_importance = estimator.feature_importances_.argsort()
                feature_importance = estimator.feature_importances_[sorted_feature_importance]
            elif model_name == 'lasso':
                sorted_feature_importance = estimator.coef_.argsort()
                feature_importance = estimator.coef_[sorted_feature_importance]
            plt.barh(self.data_X_train.columns[sorted_feature_importance], 
                    feature_importance, 
                    color='turquoise')
            plt.xlabel(model_name + " Feature Importance")
            plt.title(self.data_y_train.columns[i])
            plt.tight_layout()
        
        plt.savefig(PERMA_MODEL_RESULTS_DIR / f'{self.database_name}_{model_name}_feature_importance.png')
        
        plt.show()
        plt.clf()
        
    def plot_and_save_shap_values(self, regr, model_name):        
    
        # Only plot shap values for the top 5 important features for each estimator
        for i, estimator in enumerate(regr.estimators_):
            if model_name == 'catboost' or model_name == 'xgboost':
                explainer = shap.TreeExplainer(estimator)
                sorted_feature_importance = estimator.feature_importances_.argsort()
            elif model_name == 'lasso':
                explainer = shap.Explainer(estimator, self.data_X_train)

                sorted_feature_importance = estimator.coef_.argsort()
            shap_values = explainer.shap_values(self.data_X_train)
            
            columns_shap_values = self.data_X_train.columns[sorted_feature_importance]
            shap_values = shap_values[:, sorted_feature_importance]

            shap.summary_plot(shap_values, self.data_X_train[columns_shap_values], feature_names = columns_shap_values, show=False)
            # plt.xlabel("SHAP Values")
            plt.title(self.data_y_train.columns[i])
            plt.savefig(PERMA_MODEL_RESULTS_DIR / f'{self.database_name}_{model_name}_shap_values_{self.data_y_train.columns[i]}.png')
            plt.show()
            plt.clf()