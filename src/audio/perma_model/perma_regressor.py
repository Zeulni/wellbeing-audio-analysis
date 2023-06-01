import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import shap
import pandas as pd
import os
import multiprocessing

from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from scipy.optimize import minimize_scalar
from sklearn.base import clone

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


from math import sqrt

from src.audio.utils.constants import PERMA_MODEL_RESULTS_DIR

class PermaRegressor:
    def __init__(self, data_X_train, data_X_test, data_y_train, data_y_test, perma_feature_list, database_name) -> None:
        self.data_X_train = data_X_train
        self.data_X_test = data_X_test
        self.data_y_train = data_y_train
        self.data_y_test = data_y_test
        
        self.perma_feature_list = perma_feature_list
        
        self.database_name = database_name
        
        self.baseline_comp_dict = {"P": {"baseline": [], "prediction": []},
                                   "E": {"baseline": [], "prediction": []},
                                   "R": {"baseline": [], "prediction": []},
                                   "M": {"baseline": [], "prediction": []},
                                   "A": {"baseline": [], "prediction": []}}
        
        
        # * Ridge Model
        self.ridge_param_grid = {
            'alpha': [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
        }
        self.ridge_reg_model = Ridge()
        
        # * Lasso Model
        self.lasso_param_grid = {
            'alpha': [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
        }
        self.lasso_reg_model = Lasso()
        
        # * CatBoost Model
        self.catboost_param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [100, 200]
        }
        self.catboost_reg_model = CatBoostRegressor(loss_function='MAE' ,verbose=False, save_snapshot=False, allow_writing_files=False, train_dir=str(PERMA_MODEL_RESULTS_DIR))
        
        # * XGBoost Model
        self.xgboost_param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [100, 200]
        }
        
        # Performs better with squarrederror than absolute error
        self.xgboost_reg_model = xgb.XGBRegressor(objective='reg:squarederror')  
        
        # Create the lists with params and models
        self.model_name_list = ["Ridge", "Lasso", "XGBoost", "CatBoost"]
        self.model_param_grid_list = [self.ridge_param_grid, self.lasso_param_grid, self.xgboost_param_grid, self.catboost_param_grid]
        self.reg_model_list = [self.ridge_reg_model, self.lasso_reg_model, self.xgboost_reg_model, self.catboost_reg_model]
        
    def train_model(self, multioutput_reg_model, model_name, param_grid):        
        
        loo = LeaveOneOut() 
        rmse = "neg_root_mean_squared_error"
        grid_search = GridSearchCV(multioutput_reg_model, param_grid, cv=loo, scoring=rmse, verbose=0, n_jobs=-1, refit=rmse)
        grid_search.fit(self.data_X_train, self.data_y_train)
        
        print(model_name + " Best Hyperparameters:", grid_search.best_params_)
        print(model_name + " Best Score " + rmse + ": ", round(-grid_search.best_score_,3))
        
        multioutput_reg_model = grid_search.best_estimator_
        
        # Print baseline RMSE and MAE
        self.calc_baseline(multioutput_reg_model, model_name)
    
        # Save models dict to pickle file
        model_file_name = self.database_name + '_' + model_name + '_perma_model.pkl'
        with open(PERMA_MODEL_RESULTS_DIR / model_file_name, 'wb') as f:
            pkl.dump(multioutput_reg_model, f)
            
    def train_ind_models(self):        
            
        # Check whether best models, best model names and best params do already exist, if yes, load them and return
        best_models_file_name = os.path.join(PERMA_MODEL_RESULTS_DIR, "regression_" + self.database_name + "_best_models.pkl")
        best_models_names_file_name = os.path.join(PERMA_MODEL_RESULTS_DIR, "regression_" + self.database_name + "_best_model_names.pkl")
        best_params_file_name = os.path.join(PERMA_MODEL_RESULTS_DIR, "regression_" + self.database_name + "_best_params.pkl")
        baseline_comp_dict = os.path.join(PERMA_MODEL_RESULTS_DIR, "regression_" + self.database_name + "_baseline_comp_dict.pkl")
        if os.path.exists(best_models_file_name) and os.path.exists(best_models_names_file_name) and os.path.exists(best_params_file_name) and os.path.exists(baseline_comp_dict):
            with open(best_models_file_name, "rb") as f:
                self.best_models = pkl.load(f)
            with open(best_models_names_file_name, "rb") as f:
                self.best_models_names = pkl.load(f)
            with open(best_params_file_name, "rb") as f:
                self.best_params = pkl.load(f)
            with open(baseline_comp_dict, "rb") as f:
                self.baseline_comp_dict = pkl.load(f)
            return


        self.best_models = []
        self.best_models_names = []
        self.best_params = []
        # Loop over every pillar
        for perma_i in range(self.data_y_train.shape[1]):
            pillar_models = []
            pillar_best_scores = []
            pillar_best_params = []
            for model_i, reg_model_template in enumerate(self.reg_model_list):
            
                reg_model = clone(reg_model_template)
                param_grid = self.model_param_grid_list[model_i]
                model_name = self.model_name_list[model_i]
                #cv=LeaveOneOut()
                grid_search = GridSearchCV(reg_model, param_grid, cv=LeaveOneOut(), scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1, refit='neg_mean_absolute_error')
                grid_search.fit(self.data_X_train[self.perma_feature_list[perma_i]], self.data_y_train.iloc[:, perma_i])
                # models_trained.append(grid_search.best_estimator_)
                # self.best_models.append(grid_search.best_estimator_)
                pillar_best_params.append(grid_search.best_params_)
                pillar_models.append(grid_search.best_estimator_)
                pillar_best_scores.append(-grid_search.best_score_)
                # best_scores.append(-grid_search.best_score_)
                
                print("Best score for " + model_name + " : ", round(-grid_search.best_score_,3))
            
            # Select the best model and store in in the models list
            best_model_i = np.argmin(pillar_best_scores)
            self.best_models.append(pillar_models[best_model_i])
            self.best_models_names.append(self.model_name_list[best_model_i])
            self.best_params.append(pillar_best_params[best_model_i])
            
            self.calc_baseline_comparison(self.best_models[perma_i], self.best_models_names[perma_i], perma_i)
            
            perma_pillar = str(self.data_y_train.columns[perma_i])
            model_file_name = self.database_name + '_' + self.best_models_names[perma_i] + '_' + perma_pillar + '_perma_model.pkl'
            with open(PERMA_MODEL_RESULTS_DIR / model_file_name, 'wb') as f:
                pkl.dump(self.best_models[perma_i], f)

        # Save the best models as pickle file
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, "regression_" + self.database_name + "_best_models.pkl"), "wb") as f:
            pkl.dump(self.best_models, f)
            
        # Save the best model names as pickle file
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, "regression_" + self.database_name + "_best_model_names.pkl"), "wb") as f:
            pkl.dump(self.best_models_names, f)
            
        # Save the best params as pickle file
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, "regression_" + self.database_name + "_best_params.pkl"), "wb") as f:
            pkl.dump(self.best_params, f)
            
        # Save the self.baseline_comp_dict
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, "regression_" + self.database_name + "_baseline_comp_dict.pkl"), "wb") as f:
            pkl.dump(self.baseline_comp_dict, f)

    def train_multiple_models_per_pillar(self):
        
        self.train_ind_models()
        self.plot_and_save_feature_importance()
        self.plot_and_save_shap_values()
        self.plot_baseline_bar_plot_comparison()
        
    # define a function that measures the imbalance of the classes
    def imbalance(self, border, col):
        bins = pd.cut(col, bins=[-np.inf, border, np.inf], labels=False)
        counts = pd.value_counts(bins)
        return abs(counts[0] - counts[1])
        
    def transform_to_classification(self):
        
        for y_i, reg_model in enumerate(self.best_models):
            model_name = self.best_models_names[y_i]
        
            number_of_classes = 2
            labels = [i for i in range(number_of_classes)]
            # Based on train set, choose the borders of the bins so that the number of samples in each bin is equal
            data_y_train = self.data_y_train.iloc[:, y_i]
            # print(data_y_train.value_counts())
            # Print value counts sorted by value
            print(data_y_train.value_counts().sort_index())

            # * Quantile-Based Binning (other option: equal-width binning), not optimal for skewed distributions
            percentiles = np.linspace(0, 100, num=(number_of_classes+1))
            bin_edges = np.percentile(data_y_train, percentiles)
            
            # Add a lower bound to the first bin edge and an upper bound to the last bin edge
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            # Print the distribution of each class in the train set
            print("Distribution of classes in train set for " + str(self.data_y_train.columns[y_i]) + ":")
            print(pd.cut(data_y_train, bins=bin_edges, labels=labels).value_counts().sort_index())
        
            # Calculate the true classes for the test set (bin the PERMA scores into 3 classes)

            data_y_test = self.data_y_test.iloc[:, y_i]
            true_classes = pd.cut(data_y_test, bins=bin_edges, labels=labels)
            true_classes = true_classes.to_numpy()
            
            # Calculate the predicted classes for the test set (bin the PERMA scores into 3 classes)
            y_pred = reg_model.predict(self.data_X_test[self.perma_feature_list[y_i]])
            pred_classes = pd.cut(y_pred, bins=bin_edges, labels=labels)
            pred_classes = pred_classes.to_numpy()
            
            # Plot the confusion matrix
            conf_matrix = confusion_matrix(true_classes, pred_classes)
            print("---- Confusion matrix for " + model_name + " " + str(self.data_y_train.columns[y_i]) + " : \n", conf_matrix)
            
            # Calculate the classification report
            class_report = classification_report(true_classes, pred_classes)
            perma_pillar = str(self.data_y_train.columns[y_i])
            print("---- Classification report for " + model_name + " " + perma_pillar + " : \n", class_report)

    def calc_baseline_comparison(self, reg_model, model_name, y_i):
        
        perma_pillar = str(self.data_y_train.columns[y_i])
        data_y_train = self.data_y_train.iloc[:, y_i]
        data_y_test = self.data_y_test.iloc[:, y_i]
             
        # Select only the y_i column and then calculate the mean of the PERMA scores
        mean_perma_scores = data_y_train.mean(axis=0)
        
        y_pred_baseline = np.tile(mean_perma_scores, (len(data_y_test), 1))
        
        # Calculate the RMSE of the baseline model
        baseline_rmse = np.sqrt(mean_squared_error(data_y_test, y_pred_baseline))
        
        # Calculate the R2 score of the baseline model
        baseline_r2 = r2_score(data_y_test, y_pred_baseline)
        
        # Calculate the MAE of the baseline model
        baseline_mae = mean_absolute_error(data_y_test, y_pred_baseline)
        
        # print(model_name + " Test Set Baseline r2:", round(baseline_r2, 3))
        # print(model_name + " Test Set Baseline RMSE:", round(baseline_rmse,3))
        print (model_name + " Test Set Baseline MAE:", round(baseline_mae,3))
        self.baseline_comp_dict[perma_pillar]["baseline"].append(baseline_mae)
        
        # Using the entire dataset as test set (with reg_model) to comput R2, RMSE, MAE
        y_pred = reg_model.predict(self.data_X_test[self.perma_feature_list[y_i]])
        r2 = r2_score(data_y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(data_y_test, y_pred))
        mae = mean_absolute_error(data_y_test, y_pred)
        
        # print(model_name + " Test Set Prediction r2:", round(r2,3))
        # print(model_name + " Test Set Prediction RMSE:", round(rmse,3))
        print(model_name + " Test Set Prediction MAE:", round(mae,3))
        self.baseline_comp_dict[perma_pillar]["prediction"].append(mae)
        
    def plot_baseline_bar_plot_comparison(self):
        
        # Initialize empty lists to store the model names and average values
        perma_pillars = []
        baseline_avgs = []
        prediction_avgs = []

        # Iterate over the keys in the baseline_comp_dict dictionary to extract the model names and values
        for perma_pillar, values in self.baseline_comp_dict.items():
            perma_pillars.append(perma_pillar)
            baseline_avgs.append(np.mean(values["baseline"]))
            prediction_avgs.append(np.mean(values["prediction"]))

        # Set up the plot
        plt.rcParams['font.size'] = 16
        fig, ax = plt.subplots()
        bar_width = 0.35
        opacity = 0.8
        index = np.arange(len(perma_pillars))

        # Create the bars for the baseline values
        rects1 = ax.bar(index, baseline_avgs, bar_width,
                        alpha=opacity,
                        color='#333333',
                        label='Baseline')

        # Create the bars for the prediction values
        rects2 = ax.bar(index + bar_width, prediction_avgs, bar_width,
                        alpha=opacity,
                        color='#999999',
                        label='Prediction')
        
        # Show over each bars the name of the best model (in self.best_models_names), the names are tilted 90 degrees
        for i, rect in enumerate(rects2):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 0.4*height,
                    self.best_models_names[i],
                    ha='center', va='bottom', rotation=90, fontsize=15)
       
        # Add labels, title and legend to the plot
        ax.set_xlabel('PERMA pillars', fontsize=14)
        ax.set_ylabel('Mean absolute error', fontsize=14)
        # ax.set_title('Baseline vs. Prediction MAE by PERMA Pillar (on Test Set)', fontsize=12)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(perma_pillars, fontsize=20)
        ax.legend(loc="lower right")
        
        plt.tight_layout()
        
        plt.grid(True, alpha=0.5)
        
        # Save the plot in PERMA_MODEL_RESULTS_DIR
        filename = os.path.join(PERMA_MODEL_RESULTS_DIR, "regression_test_set_eval_" + self.database_name + ".png")
        plt.savefig(filename, dpi=600)
        
        # Adjust plot to the left and right
        plt.subplots_adjust(left=0.20, right=0.96, top=0.9, wspace=1.8, hspace=0.3)
        
        # Show the plot
        plt.show()
        
    def plot_baseline_box_plot_comparison(self):
        
        # Initialize empty list to store the model names and prediction values
        models = []
        prediction_values = []

        # Iterate over the keys in the baseline_comp_dict dictionary to extract the model names and prediction values
        for model, values in self.baseline_comp_dict.items():
            models.append(model)
            prediction_values.append(values["prediction"])

        # Set up the plot
        plt.rcParams['font.size'] = 16
        fig, ax = plt.subplots()

        # Create the box plot for the prediction values
        ax.boxplot(prediction_values)

        # Add labels, title and legend to the plot
        ax.set_xlabel('Models')
        ax.set_ylabel('Prediction Values')
        ax.set_title('Prediction Values by Model')
        ax.set_xticklabels(models, fontsize=20)

        # Show the plot
        plt.show()
    
    def catboost_train(self):
        
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
        
    def plot_and_save_feature_importance(self):
        
        # Close all current plots
        plt.close('all')
        
        # Define the size of the plot
        plt.figure(figsize=(16, 6))
            
        # Loop over every estimator and plot the feature importance as a subplot
        for i, reg_model in enumerate(self.best_models):
            model_name = self.best_models_names[i]
            plt.subplot(2, 3, i+1)
            if model_name == 'CatBoost' or model_name == 'XGBoost':
                # sorted_feature_importance = reg_model.feature_importances_.argsort()
                # feature_importance = reg_model.feature_importances_[sorted_feature_importance]
                feature_importance = reg_model.feature_importances_
                feature_names = reg_model.feature_names_
            elif model_name == 'Lasso' or model_name == 'Ridge':
                # sorted_feature_importance = reg_model.coef_.argsort()
                # feature_importance = reg_model.coef_[sorted_feature_importance]
                feature_importance = reg_model.coef_
                feature_names = reg_model.feature_names_in_
            
            # Sort the feature importance and feature names in ascending order
            sorted_feature_importance = feature_importance.argsort()
            feature_importance = feature_importance[sorted_feature_importance]
            feature_names = np.array(feature_names)[sorted_feature_importance]
            
            plt.barh(feature_names, 
                    feature_importance, 
                    color='#666666')
            plt.xlabel(model_name + " feature importance")
            perma_pillar = str(self.data_y_train.columns[i])
            plt.title(perma_pillar, fontsize=20, fontweight='bold')
            
            # Add gridlines along the y-axis
            plt.grid(True, axis='x', alpha=0.5)
            
        # plt.tight_layout()    
        plt.subplots_adjust(left=0.21, right=0.98, top=0.9, wspace=1.8, hspace=0.5)
        plt.savefig(PERMA_MODEL_RESULTS_DIR / f'{self.database_name}_regression_feature_importance_overview.png', dpi=600)
        plt.show()
        plt.clf()
        
    def plot_and_save_shap_values(self):        
    
        # Only plot shap values for the top 5 important features for each estimator
        for i, reg_model in enumerate(self.best_models):
            model_name = self.best_models_names[i]
            if model_name == 'CatBoost' or model_name == 'XGBoost':
                explainer = shap.TreeExplainer(reg_model)
                feature_names = reg_model.feature_names_
            elif model_name == 'Lasso' or model_name == 'Ridge':
                explainer = shap.Explainer(reg_model, self.data_X_train[self.perma_feature_list[i]])
                feature_names = reg_model.feature_names_in_
                
            shap_values = explainer.shap_values(self.data_X_train[self.perma_feature_list[i]])
            # feature_names = [" "]

            shap.summary_plot(shap_values, self.data_X_train[self.perma_feature_list[i]], feature_names = feature_names, show=False)
            plt.xlabel("SHAP Values")
            perma_pillar = str(self.data_y_train.columns[i])
            # plt.title(perma_pillar, fontsize=20, fontweight='bold')
            plt.savefig(PERMA_MODEL_RESULTS_DIR / f'{self.database_name}_{model_name}_regression_shap_values_{perma_pillar}.png', dpi=600)
            # plt.show()
            plt.clf()
            
    def calc_rel_baseline_comp(self):

        baseline_comp_filename = os.path.join(PERMA_MODEL_RESULTS_DIR, "regression_" + self.database_name + "_baseline_comp_dict.pkl")
        
        with open(baseline_comp_filename, "rb") as f:
            baseline_comp_dict = pkl.load(f)
        
        # Calculate for each dictionary the ratio of the prediction to the baseline and store it in a new dictionary
        baseline_comp_dict_ratio = {}

        for perma_pillar, values in baseline_comp_dict.items():
            baseline_comp_dict_ratio[perma_pillar] = (values["baseline"][0]-values["prediction"][0])/values["baseline"][0]*100
            
        overall_improvement = np.mean(list(baseline_comp_dict_ratio.values()))
            
        print("test")